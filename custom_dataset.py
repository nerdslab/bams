import os
import numpy as np
import argparse
from datetime import datetime


import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from bams.data import KeypointsDataset
from bams.models import BAMS
from bams import HoALoss


def load_data(path):
    keypoints = ...
    return keypoints


def train(model, device, loader, optimizer, criterion, writer, step, log_every_step):
    model.train()

    for data in tqdm(loader, position=1, leave=False):
        # todo convert to float
        input = data["input"].float().to(device)  # (B, N, L)
        target = data["target_hist"].float().to(device)
        ignore_weights = data["ignore_weights"].to(device)

        # forward pass
        optimizer.zero_grad()
        embs, hoa_pred, byol_preds = model(input)

        # prediction task
        hoa_loss = criterion(target, hoa_pred, ignore_weights)

        # contrastive loss: short term
        batch_size, sequence_length, emb_dim = embs["short_term"].size()
        skip_frames, delta = 60, 5
        view_1_id = (
            torch.randint(sequence_length - skip_frames - delta, (batch_size,))
            + skip_frames
        )
        view_2_id = torch.randint(delta + 1, (batch_size,)) + view_1_id
        view_2_id = torch.clip(view_2_id, 0, sequence_length)

        view_1 = byol_preds["short_term"][torch.arange(batch_size), view_1_id]
        view_2 = embs["short_term"][torch.arange(batch_size), view_2_id]

        byol_loss_short_term = (
            1 - F.cosine_similarity(view_1, view_2.clone().detach(), dim=-1).mean()
        )

        # contrastive loss: long term
        batch_size, sequence_length, emb_dim = embs["long_term"].size()
        skip_frames = 100
        view_1_id = (
            torch.randint(sequence_length - skip_frames, (batch_size,)) + skip_frames
        )
        view_2_id = (
            torch.randint(sequence_length - skip_frames, (batch_size,)) + skip_frames
        )

        view_1 = byol_preds["long_term"][torch.arange(batch_size), view_1_id]
        view_2 = embs["long_term"][torch.arange(batch_size), view_2_id]

        byol_loss_long_term = (
            1 - F.cosine_similarity(view_1, view_2.clone().detach(), dim=-1).mean()
        )

        # backprop
        loss = 5e2 * hoa_loss + 0.5 * byol_loss_short_term + 0.5 * byol_loss_long_term

        loss.backward()
        optimizer.step()

        step += 1
        if step % log_every_step == 0:
            writer.add_scalar("train/hoa_loss", hoa_loss.item(), step)
            writer.add_scalar(
                "train/byol_loss_short_term", byol_loss_short_term.item(), step
            )
            writer.add_scalar(
                "train/byol_loss_long_term", byol_loss_long_term.item(), step
            )
            writer.add_scalar("train/total_loss", loss.item(), step)

    return step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/mabe")
    parser.add_argument("--cache_path", type=str, default="./data/mabe/custom_dataset")
    parser.add_argument("--hoa_bins", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--log_every_step", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    keypoints = load_data(args.data_root)

    dataset = KeypointsDataset(
        keypoints=keypoints,
        hoa_bins=args.hoa_bins,
        cache_path=args.cache_path,
        cache=False,
    )
    print("Number of sequences:", len(dataset))

    # prepare dataloaders
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # build model
    model = BAMS(
        input_size=dataset.input_size,
        short_term=dict(num_channels=(64, 64, 64, 64), kernel_size=3),
        long_term=dict(num_channels=(64, 64, 64, 64, 64), kernel_size=3, dilation=4),
        predictor=dict(
            hidden_layers=(-1, 256, 512, 512, dataset.target_size * args.hoa_bins)
        ),
    ).to(device)

    print(model)

    model_name = f"bams-custom-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    writer = SummaryWriter("runs/" + model_name)

    main_params = [p for name, p in model.named_parameters() if "byol" not in name]
    byol_params = list(model.byol_predictors.parameters())

    optimizer = optim.AdamW(
        [{"params": main_params}, {"params": byol_params, "lr": args.lr * 10}],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.1)
    criterion = HoALoss(hoa_bins=args.hoa_bins, skip_frames=100)

    step = 0
    for epoch in tqdm(range(1, args.epochs + 1), position=0):
        step = train(
            model,
            device,
            train_loader,
            optimizer,
            criterion,
            writer,
            step,
            args.log_every_step,
        )
        scheduler.step()

        if epoch % 100 == 0:
            torch.save(model.state_dict(), model_name + ".pt")


if __name__ == "__main__":
    main()
