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

from bams.data import Dataset
from bams.data.utils import diff, to_polar_coordinates, angle_clip
from bams.models import BAMS
from bams import HoALoss


#############
# Load data #
#############
def load_mice_triplet(path):
    # load raw train data (with annotations for 2 tasks)
    data_train = np.load(
        os.path.join(path, "mouse_triplet_train.npy"), allow_pickle=True
    ).item()
    sequence_ids_train, sequence_data_train = zip(*data_train["sequences"].items())
    keypoints_train = np.stack([data["keypoints"] for data in sequence_data_train])

    # load submission data (no annoations)
    data_submission = np.load(
        os.path.join(path, "mouse_triplet_test.npy"), allow_pickle=True
    ).item()
    sequence_ids_submission, sequence_data_submission = zip(
        *data_submission["sequences"].items()
    )
    keypoints_submission = np.stack(
        [data["keypoints"] for data in sequence_data_submission]
    )

    # concatenate train and submission data
    sequence_ids = np.concatenate([sequence_ids_train, sequence_ids_submission], axis=0)
    keypoints = np.concatenate([keypoints_train, keypoints_submission], axis=0)

    split_mask = np.ones(len(sequence_ids), dtype=bool)
    split_mask[-len(sequence_ids_submission) :] = False

    # treat each mouse independently, keep track of which video each mouse came from
    num_samples, sequence_length, num_mice, num_keypoints, _ = keypoints.shape
    keypoints = keypoints.transpose((0, 2, 1, 3, 4))
    keypoints = keypoints.reshape((-1, sequence_length, num_keypoints, 2))
    batch = np.repeat(np.arange(num_samples), num_mice)

    return keypoints, split_mask, batch


################
# Process data #
################
def mouse_feature_extractor(keypoints, noise_thresh=3e-3):
    # compute state features
    # body part 1: head, keypoints 0, 1, 2, 3
    head_center = keypoints[..., 3, :]
    head_orientation = np.arctan2(
        keypoints[..., 0, 1] - keypoints[..., 3, 1],
        keypoints[..., 0, 0] - keypoints[..., 3, 0],
    )

    # body part 2: forepaws, keypoints 3, 4, 5
    # use keypoint 3 as center
    left_forepaw = keypoints[..., 4, :] - keypoints[..., 3, :]
    right_forepaw = keypoints[..., 5, :] - keypoints[..., 3, :]

    left_forepaw_r, left_forepaw_theta = to_polar_coordinates(left_forepaw)
    right_forepaw_r, right_forepaw_theta = to_polar_coordinates(right_forepaw)
    forepaws_theta = angle_clip(right_forepaw_theta - left_forepaw_theta)

    # connection body parts 2-3
    spine = keypoints[..., 6, :] - keypoints[..., 3, :]
    spine_r, spine_theta = to_polar_coordinates(spine)

    # body part 3: bottom, keypoints 6, 7, 8, 9
    bottom_center = keypoints[..., 6, :]
    # center
    bottom = keypoints[..., 7:, :] - bottom_center[..., np.newaxis, :]
    bottom_orientation = np.arctan2(
        keypoints[..., 6, 1] - keypoints[..., 9, 1],
        keypoints[..., 6, 0] - keypoints[..., 9, 0],
    )

    bottom_rotation = np.array(
        [
            [np.cos(-bottom_orientation), -np.sin(-bottom_orientation)],
            [np.sin(-bottom_orientation), np.cos(-bottom_orientation)],
        ]
    )
    # rotate
    bottom = np.einsum("ijkp,lpij->ijkl", bottom, bottom_rotation)

    left_hindpaw_r, left_hindpaw_theta = to_polar_coordinates(bottom[..., 0, :])
    left_hindpaw_theta = left_hindpaw_theta
    right_hindpaw_r, right_hindpaw_theta = to_polar_coordinates(bottom[..., 1, :])
    right_hindpaw_theta = right_hindpaw_theta
    center_to_tail_r, _ = to_polar_coordinates(bottom[..., 2, :])

    _, tail_theta_1 = to_polar_coordinates(bottom[..., 3, :] - bottom[..., 2, :])
    tail_theta_1 = tail_theta_1
    _, tail_theta_2 = to_polar_coordinates(bottom[..., 4, :] - bottom[..., 3, :])
    tail_theta_2 = tail_theta_2

    # compute action features
    ### body part 1: head
    head_vx = diff(head_center[..., 0])
    head_vy = diff(head_center[..., 0])
    head_vr, head_vtheta = to_polar_coordinates(np.stack([head_vx, head_vy], axis=-1))
    head_vtheta[head_vr < noise_thresh] = 0.0
    head_vr[head_vr < noise_thresh] = 0.0
    head_dvtheta = angle_clip(diff(head_vtheta))
    # orientation
    head_orientation_dtheta = angle_clip(diff(head_orientation))
    ### body part 2: forepaws
    # left forepaw
    left_forepaw_dr = diff(left_forepaw_r)
    left_forepaw_dtheta = angle_clip(diff(left_forepaw_theta))
    # right forepaw
    right_forepaw_dr = diff(left_forepaw_r)
    right_forepaw_dtheta = angle_clip(diff(right_forepaw_theta))
    # angle between forepaws
    forepaws_dtheta = angle_clip(diff(forepaws_theta))
    # body part 3: bottom
    # velocity
    bottom_vx = diff(bottom_center[..., 0])
    bottom_vy = diff(bottom_center[..., 1])
    bottom_vr, bottom_vtheta = to_polar_coordinates(
        np.stack([bottom_vx, bottom_vy], axis=-1)
    )
    bottom_vtheta[bottom_vr < noise_thresh] = 0.0
    bottom_vr[bottom_vr < noise_thresh] = 0.0
    bottom_dvtheta = angle_clip(diff(bottom_vtheta))
    # orientation
    bottom_orientation_dtheta = angle_clip(diff(bottom_orientation))
    # left hindpaw
    left_hindpaw_dr = diff(left_hindpaw_r)
    left_hindpaw_dtheta = angle_clip(diff(left_hindpaw_theta))
    # right hindpaw
    right_hindpaw_dr = diff(right_hindpaw_r)
    right_hindpaw_dtheta = angle_clip(diff(right_hindpaw_theta))
    # body part 4: tail
    tail_dtheta_1 = angle_clip(diff(tail_theta_1))
    tail_dtheta_2 = angle_clip(diff(tail_theta_2))
    # connections between body parts
    center_to_tail_dr = diff(center_to_tail_r)
    spine_dr = diff(spine_r)
    spine_dtheta = angle_clip(diff(spine_theta))

    ignore_frames = np.any(keypoints[..., 0] == 0, axis=-1)
    ignore_frames[:, 1:] = np.logical_or(ignore_frames[:, 1:], ignore_frames[:, :-1])

    input_features = np.stack(
        [
            head_center[..., 0],
            head_center[..., 1],
            np.cos(head_orientation),
            np.sin(head_orientation),
            left_forepaw_r,
            np.cos(left_forepaw_theta),
            np.sin(left_forepaw_theta),
            right_forepaw_r,
            np.cos(right_forepaw_theta),
            np.sin(right_forepaw_theta),
            np.cos(forepaws_theta),
            np.sin(forepaws_theta),
            bottom_center[..., 0],
            bottom_center[..., 1],
            np.cos(bottom_orientation),
            np.sin(bottom_orientation),
            left_hindpaw_r,
            np.cos(left_hindpaw_theta),
            np.sin(left_hindpaw_theta),
            right_hindpaw_r,
            np.cos(right_hindpaw_theta),
            np.sin(right_hindpaw_theta),
            center_to_tail_r,
            np.cos(tail_theta_1),
            np.sin(tail_theta_1),
            np.cos(tail_theta_2),
            np.sin(tail_theta_2),
            spine_r,
            np.cos(spine_theta),
            np.sin(spine_theta),
            head_vr,
            np.cos(head_vtheta),
            np.sin(head_vtheta),
            np.cos(head_dvtheta),
            np.sin(head_dvtheta),
            np.cos(head_orientation_dtheta),
            np.sin(head_orientation_dtheta),
            left_forepaw_dr,
            np.cos(left_forepaw_dtheta),
            np.sin(left_forepaw_dtheta),
            right_forepaw_dr,
            np.cos(right_forepaw_dtheta),
            np.sin(right_forepaw_dtheta),
            np.cos(forepaws_dtheta),
            np.sin(forepaws_dtheta),
            bottom_vr,
            np.cos(bottom_vtheta),
            np.sin(bottom_vtheta),
            np.cos(bottom_dvtheta),
            np.sin(bottom_dvtheta),
            np.cos(bottom_orientation_dtheta),
            np.sin(bottom_orientation_dtheta),
            left_hindpaw_dr,
            np.cos(left_hindpaw_dtheta),
            np.sin(left_hindpaw_dtheta),
            right_hindpaw_dr,
            np.cos(right_hindpaw_dtheta),
            np.sin(right_hindpaw_dtheta),
            np.cos(tail_dtheta_1),
            np.sin(tail_dtheta_1),
            np.cos(tail_dtheta_2),
            np.sin(tail_dtheta_2),
            center_to_tail_dr,
            spine_dr,
            np.cos(spine_dtheta),
            np.sin(spine_dtheta),
            ignore_frames,
        ],
        axis=-1,
    )

    target_feats = np.stack(
        [
            head_vr,
            head_vtheta,
            head_dvtheta,
            head_orientation_dtheta,
            bottom_vr,
            bottom_vtheta,
            bottom_dvtheta,
            bottom_orientation_dtheta,
            spine_dr,
        ],
        axis=-1,
    )

    return input_features, target_feats, ignore_frames


#################
# Training loop #
#################
def train_loop(
    model, device, loader, optimizer, criterion, writer, step, log_every_step
):
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
    parser.add_argument(
        "--job",
        default="train",
        const="train",
        nargs="?",
        choices=["train", "compute_representations"],
        help="select task",
    )
    parser.add_argument("--data_root", type=str, default="./data/mabe")
    parser.add_argument("--cache_path", type=str, default="./data/mabe/mouse_triplet")
    parser.add_argument("--hoa_bins", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--log_every_step", type=int, default=50)
    parser.add_argument("--ckpt_path", type=str, default=None)
    args = parser.parse_args()

    if args.job == "train":
        train(args)
    elif args.job == "compute_representations":
        compute_representations(args)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    if not Dataset.cache_is_available(args.cache_path, args.hoa_bins):
        print("Processing data...")
        keypoints, split_mask, batch = load_mice_triplet(args.data_root)
        input_feats, target_feats, ignore_frames = mouse_feature_extractor(keypoints)
    else:
        print("No need to process data")
        input_feats = target_feats = ignore_frames = None

    dataset = Dataset(
        input_feats=input_feats,
        target_feats=target_feats,
        ignore_frames=ignore_frames,
        cache_path=args.cache_path,
        cache=True,
        hoa_bins=args.hoa_bins,
        hoa_window=30,
    )

    print("Number of sequences:", len(dataset))

    # prepare dataloaders
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # build model
    model = BAMS(
        input_size=dataset.input_size,
        short_term=dict(num_channels=(64, 64, 32, 32), kernel_size=3),
        long_term=dict(num_channels=(64, 64, 64, 32, 32), kernel_size=3, dilation=4),
        predictor=dict(
            hidden_layers=(-1, 256, 512, 512, dataset.target_size * args.hoa_bins),
        ),  # frame rate = 30, 6 steps = 200ms
    ).to(device)

    model_name = f"bams-mouse-triplet-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    writer = SummaryWriter("runs/" + model_name)

    main_params = [p for name, p in model.named_parameters() if "byol" not in name]
    byol_params = list(model.byol_predictors.parameters())

    optimizer = optim.AdamW(
        [{"params": main_params}, {"params": byol_params, "lr": args.lr * 10}],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.1)
    criterion = HoALoss(hoa_bins=args.hoa_bins, skip_frames=60)

    step = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        step = train_loop(
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


def compute_representations(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    keypoints, split_mask, batch = load_mice_triplet(args.data_root)

    # dataset
    if not Dataset.cache_is_available(args.cache_path, args.hoa_bins):
        print("Processing data...")
        input_feats, target_feats, ignore_frames = mouse_feature_extractor(keypoints)
    else:
        print("No need to process data")
        input_feats = target_feats = ignore_frames = None

    # only use

    dataset = Dataset(
        input_feats=input_feats,
        target_feats=target_feats,
        ignore_frames=ignore_frames,
        cache_path=args.cache_path,
        hoa_bins=args.hoa_bins,
        hoa_window=30,
    )

    print("Number of sequences:", len(dataset))

    # build model
    model = BAMS(
        input_size=dataset.input_size,
        short_term=dict(num_channels=(64, 64, 32, 32), kernel_size=3),
        long_term=dict(num_channels=(64, 64, 64, 32, 32), kernel_size=3, dilation=4),
        predictor=dict(
            hidden_layers=(-1, 256, 512, 512, dataset.target_size * args.hoa_bins),
        ),  # frame rate = 30, 6 steps = 200ms
    ).to(device)

    if args.ckpt_path is None:
        raise ValueError("Please specify a checkpoint path")

    # load checkpoint
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model.eval()

    loader = DataLoader(
        dataset,
        shuffle=False,
        drop_last=False,
        batch_size=32,
        num_workers=16,
        pin_memory=True,
    )

    # compute representations
    short_term_emb, long_term_emb = [], []

    for data in loader:
        input = data["input"].float().to(device)  # (B, N, L)

        with torch.inference_mode():
            embs, _, _ = model(input)

            short_term_emb.append(embs["short_term"].detach().cpu())
            long_term_emb.append(embs["long_term"].detach().cpu())

    short_term_emb = torch.cat(short_term_emb)
    long_term_emb = torch.cat(long_term_emb)

    embs = torch.cat([short_term_emb, long_term_emb], dim=2)

    # the learned representations are at the individual mouse level, we want to compute
    # the mouse triplet-level representation
    # embs: (B, L, N)
    batch_size, seq_len, num_feats = embs.size()
    embs = embs.reshape(-1, 3, seq_len, num_feats)

    embs_mean = embs.mean(1)
    embs_max = embs.max(1).values
    embs_min = embs.min(1).values

    embs = torch.cat([embs_mean, embs_max - embs_min], dim=-1)

    # normalize embeddings
    mean, std = embs.mean(0, keepdim=True), embs.std(0, unbiased=False, keepdim=True)
    embs = (embs - mean) / std

    frame_number_map = np.load(
        os.path.join(args.data_root, "mouse_triplet_frame_number_map.npy"),
        allow_pickle=True,
    ).item()

    # only take submission frames
    embs = embs.numpy()[~split_mask].reshape(-1, embs.shape[-1])

    submission_dict = dict(
        frame_number_map=frame_number_map,
        embeddings=embs,
    )

    model_name = os.path.splitext(os.path.basename(args.ckpt_path))[0]
    np.save(f"{model_name}_submission.npy", submission_dict)


if __name__ == "__main__":
    main()
