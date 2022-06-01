import argparse
import os
from math import ceil

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models.convnext import LayerNorm2d
from tqdm import tqdm

from augment import TrivialAugmentWide
from dataset import FgvcAircraftDataset
from models import convnext_t_encoder
from simclr import SimCLR
from utils import AverageMeter, L2SP_ConvNext, ModelEma, makedirs


def train(args):
    makedirs(args.exp_path)
    device = args.device
    use_ema = args.ema

    # Model def
    encoder = convnext_t_encoder()
    immediate = SimCLR(encoder, latent_dim=128, hidden_dim=1024, t=0.1, eps=1e-6)
    immediate.load_state_dict(torch.load(args.weight))
    del immediate
    model = encoder
    model.classifier = nn.Sequential(
        LayerNorm2d(768, elementwise_affine=True, eps=1e-6),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(768, 100),
    )
    model = model.to(device)
    model.train()
    if use_ema:
        ema = ModelEma(model, decay=0.9999)

    # Train dataset and loader
    augment_train = T.Compose([
        T.Resize((300, 300)),
        T.RandomHorizontalFlip(),
        TrivialAugmentWide(),
        T.ToTensor(),
        # T.Normalize(mean=[0.4796, 0.5107, 0.5341], std=[0.1957, 0.1945, 0.2162])
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_ds = FgvcAircraftDataset(
        meta_path='./fgvc-aircraft-2013b/data/images_variant_trainval.txt',
        image_path='./fgvc-aircraft-2013b/data/images',
        transforms=augment_train,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Optimizer with L2SP
    model.zero_grad()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,
        momentum=0.9,
        nesterov=True
    )
    l2sp = L2SP_ConvNext(model, alpha=2e-4, beta=1e-5)

    # Scheduler
    estimated_steps = int(ceil(len(train_loader) * args.epochs / args.grad_acc_steps))
    scheduler = CosineAnnealingLR(optimizer, estimated_steps)

    # Grad scaler for AMP
    scaler = torch.cuda.amp.GradScaler(init_scale=2**14) # Prevents early overflow

    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    # wandb
    if args.wandb:
        import wandb
        wandb.init(project="fgvc-aircraft", entity="toduck15hl")
        wandb.config.update(args, allow_val_change=True)

    # Training loop
    step = 0
    for _ in tqdm(range(args.epochs)):
        ep_loss = AverageMeter()
        for x, y in tqdm(train_loader):
            step += 1
            with torch.cuda.amp.autocast():
                x = x.to(device)
                y = y.to(device)
                pred_S = model(x)
                loss_S = criterion(pred_S, y)/args.grad_acc_steps
                scaler.scale(loss_S).backward()
                ep_loss.update(loss_S.item())

                if step % args.grad_acc_steps == 0:
                    l2sp(model)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    model.zero_grad(set_to_none=True)
        if use_ema:
            ema.update(model)
        if wandb:
            log_info = {'train_loss': ep_loss.avg}
            wandb.log(log_info)
    
    if use_ema:
        torch.save(ema.module.state_dict(), os.path.join(args.exp_path, "ema.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(args.exp_path, "finetune.pth"))

    return model
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--exp-path', type=str, default='./exp/finetune')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--grad-acc-steps', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--ema', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _ = train(args)
