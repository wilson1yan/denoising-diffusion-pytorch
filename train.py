import os

import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from torchvision.utils import make_grid
from tensorfn import load_arg_config, load_wandb
from tensorfn import distributed as dist
from tensorfn.optim import lr_scheduler
from tqdm import tqdm

from model import UNet
from diffusion import GaussianDiffusion, make_beta_schedule
from dataset import MultiResolutionDataset, get_dataset
from config import DiffusionConfig


def sample_data(loader):
    loader_iter = iter(loader)
    epoch = 0

    while True:
        try:
            yield epoch, next(loader_iter)

        except StopIteration:
            epoch += 1
            loader_iter = iter(loader)

            yield epoch, next(loader_iter)


def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


@torch.no_grad()
def p_sample_loop(self, model, noise, device, noise_fn=torch.randn):
    img = noise
    if dist.is_primary():
        pbar = tqdm(total=self.num_timesteps)
    for i in reversed(range(self.num_timesteps)):
        img = self.p_sample(
            model,
            img,
            torch.full((img.shape[0],), i, dtype=torch.int64).to(device),
            noise_fn=noise_fn,
        )
        if dist.is_primary():
            pbar.update(1)
    if dist.is_primary():
        pbar.close()
    return img


def train(conf, loader, model, ema, diffusion, optimizer, scheduler, device, wandb):
    loader = sample_data(loader)

    pbar = range(conf.training.n_iter + 1)

    if dist.is_primary():
        pbar = tqdm(pbar, dynamic_ncols=True)

    for i in pbar:
        epoch, img = next(loader)
        img = img.to(device)
        time = torch.randint(
            0,
            conf.diffusion.beta_schedule["n_timestep"],
            (img.shape[0],),
            device=device,
        )
        loss = diffusion.p_loss(model, img, time)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        scheduler.step()
        optimizer.step()

        accumulate(
            ema, model.module, 0 if i < conf.training.scheduler.warmup else 0.9999
        )

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_description(
                f"epoch: {epoch}; loss: {loss.item():.4f}; lr: {lr:.5f}"
            )

            if wandb is not None and i % conf.evaluate.log_every == 0:
                wandb.log({"epoch": epoch, "loss": loss.item(), "lr": lr}, step=i)

            if i % conf.evaluate.save_every == 0:
                if conf.distributed:
                    model_module = model.module

                else:
                    model_module = model

                torch.save(
                    {
                        "model": model_module.state_dict(),
                        "ema": ema.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "conf": conf,
                    },
                    f"checkpoint/diffusion_{str(i).zfill(6)}.pt",
                )
            
        if i % conf.evaluate.viz_every == 0:
            img_shape = (3, conf.dataset.resolution, conf.dataset.resolution)
            noise = torch.randn(64 // dist.get_world_size(), 
                                *img_shape, dtype=torch.float32).to(device)
            samples = p_sample_loop(diffusion, ema, noise, device).cpu()
            samples = torch.cat(dist.all_gather(samples), dim=0)
            samples = torch.clamp(samples, -1, 1) * 0.5 + 0.5
            samples = make_grid(samples, nrow=8)
            if dist.is_primary():
                wandb.log({"samples": wandb.Image(samples)}, step=i)


def compute_total_params(model):
    return sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])


def main(conf):
    wandb = None
    if dist.is_primary() and conf.evaluate.wandb:
        wandb = load_wandb()
        wandb.init(project="denoising diffusion")

    device = "cuda"
    beta_schedule = "linear"

    conf.distributed = dist.get_world_size() > 1

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    train_set = get_dataset(conf, transform)
    train_sampler = dist.data_sampler(
        train_set, shuffle=True, distributed=conf.distributed
    )
    train_loader = conf.training.dataloader.make(train_set, sampler=train_sampler)

    model = conf.model.make()
    model = model.to(device)
    ema = conf.model.make()
    ema = ema.to(device)

    if dist.is_primary():
        print('total_params:', compute_total_params(model))

    if conf.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = conf.training.optimizer.make(model.parameters())
    scheduler = conf.training.scheduler.make(optimizer)

    if conf.ckpt is not None:
        ckpt = torch.load(conf.ckpt, map_location=lambda storage, loc: storage)

        if conf.distributed:
            model.module.load_state_dict(ckpt["model"])

        else:
            model.load_state_dict(ckpt["model"])

        ema.load_state_dict(ckpt["ema"])

    betas = conf.diffusion.beta_schedule.make()
    diffusion = GaussianDiffusion(betas).to(device)

    train(
        conf, train_loader, model, ema, diffusion, optimizer, scheduler, device, wandb
    )


if __name__ == "__main__":
    conf = load_arg_config(DiffusionConfig)

    dist.launch(
        main, conf.n_gpu, conf.n_machine, conf.machine_rank, conf.dist_url, args=(conf,)
    )
