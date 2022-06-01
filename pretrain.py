import argparse
import os

import torch
import torchvision.transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import FgvcAircraftSimCLR
from models import convnext_t_encoder
from simclr import SimCLR, nt_xent_loss
from utils import AverageMeter, makedirs, split_parameters


def train(args):
    makedirs(args.exp_path)
    device = args.device
    
    # Model
    encoder = convnext_t_encoder(pretrained=False)
    model = SimCLR(encoder, latent_dim=128, hidden_dim=1024, t=0.1, eps=1e-6)
    model = model.to(device)
    model.train()

    # Augmentation policy described in the paper
    augment = T.Compose([
        T.RandomResizedCrop((300, 300)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(21)], p=0.5),
        T.ToTensor(),
        # T.Normalize(mean=[0.4796, 0.5107, 0.5341], std=[0.1957, 0.1945, 0.2162]) # FGVC stats
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
    ])

    # Dataset and loader
    train_ds = FgvcAircraftSimCLR(
        meta_path='./fgvc-aircraft-2013b/data/images_family_trainval.txt',
        image_path='./fgvc-aircraft-2013b/data/images',
        transforms=augment,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Optimizer
    decay_params, no_decay_params = split_parameters(model)
    optimizer = torch.optim.SGD(
        [
            {'params': decay_params, 'weight_decay': 1e-6},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=1e-3,
        momentum=0.9,
        nesterov=True,
    )

    # Scheduler
    estimated_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=estimated_steps)

    # Loss
    criterion = nt_xent_loss

    # Grad scaler for AMP
    scaler = torch.cuda.amp.GradScaler(init_scale=2**14) # Prevents early overflow

    # Wandb
    if args.wandb:
        import wandb
        wandb.init(project="fgvc-aircraft", entity="toduck15hl")
        wandb.config.update(args, allow_val_change=True)

    # Training loop
    for ep in tqdm(range(args.epochs)):
        epoch_loss = AverageMeter()
        for x1, x2 in tqdm(train_loader):
            x1, x2 = x1.to(device), x2.to(device)
            with torch.cuda.amp.autocast():
                z1 = model(x1)
                z2 = model(x2)
                loss = criterion(z1, z2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            model.zero_grad(set_to_none=True)    
            epoch_loss.update(loss.item())

        if args.wandb:
            log_info = {'loss': epoch_loss.avg}
            wandb.log(log_info)

    torch.save(model.state_dict(), os.path.join(args.exp_path, "simclr.pth"))

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--exp-path', type=str, default='./exp/simclr')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _ = train(args)
