from functools import cached_property
import os
import numpy as np
from tqdm import tqdm


from . import CachedDataset, diff


class Dataset(CachedDataset):
    r"""Base class for all datasets.

    Args:
        root (str): Root directory of dataset.
        keypoints (np.ndarray): Array of shape (num_sequences, sequence_len,
            num_feats). Use np.nan for missing values or padding frames.
        annotations (dict(str, np.ndarray)): Dictionary of annotations, where each
            value is an array of shape (num_sequences, sequence_len).
        name (str): Name of the dataset.
        hoa_bins (int): Number of bins for the histograms of actions.
        hoa_window (int): Window size for the histograms of actions.
        force_process (bool): Whether to force process the dataset.
    """

    def __init__(
        self,
        input_feats,
        target_feats,
        ignore_frames,
        *,
        hoa_bins=32,
        hoa_window=30,
        cache_path=None,
        cache=True,
    ):
        self.input_feats = input_feats
        self.target_feats = target_feats
        self.ignore_frames = ignore_frames

        assert hoa_bins <= 255, "n_bins must be less than 256, got {}.".format(hoa_bins)
        self.hoa_bins = hoa_bins
        assert hoa_window <= 255, "hoa_window must be less than 256, got {}.".format(
            hoa_window
        )
        self.hoa_window = hoa_window
        cache_path = "./data/tmp" if cache_path is None else cache_path
        cache_path = cache_path + f"_bins{self.hoa_bins}.pkl"

        super().__init__(cache_path, cache)

    @staticmethod
    def cache_is_available(cache_path, hoa_bins):
        return os.path.exists(cache_path + f"_bins{hoa_bins}.pkl")

    def process(self):
        # quantize the target features in order to create the histogram of actions
        bins = np.zeros(
            (self.hoa_bins - 1, self.target_feats.shape[-1]), dtype=np.float32
        )
        quantized_target_feats = np.zeros_like(self.target_feats, dtype=np.uint8)

        # pre-compute histogram of actions for target features
        num_feats = self.target_feats.shape[2]
        for i in tqdm(range(num_feats)):
            # find the range of values (low, high) for each feature
            feat = self.target_feats[..., i].flatten() 
            feat = feat[~np.isnan(feat)]
            feat = feat[np.abs(feat) > 0.1]
            low, high = np.nanpercentile(feat, [0.5, 99.5])

            # compute histogram
            bins[..., i] = np.linspace(low, high, self.hoa_bins - 1)
            quantized_target_feats[..., i] = np.digitize(
                self.target_feats[..., i], bins[..., i]
            ).astype(np.uint8)

            # normalize
            self.target_feats[..., i] = self.target_feats[..., i] / np.max(
                [np.abs(low), np.abs(high)]
            )

        # normalize input features
        for i in range(self.input_feats.shape[2]):
            # z-score
            self.input_feats[..., i] = self.input_feats[..., i] / np.nanmax(
                np.abs(self.input_feats[..., i])
            )

        data = dict(
            input_feats=self.input_feats,
            target_feats=self.target_feats,
            quantized_target_feats=quantized_target_feats,
            ignore_frames=self.ignore_frames,
        )
        return data

    def __getitem__(self, item):
        # make histogram of actions
        quantized_target_feat = self.quantized_target_feats[
            item
        ]  # shape (sequence_len, num_feats)
        ignore_frames = self.ignore_frames[item]  # shape (sequence_len,)

        rows, cols = np.indices(quantized_target_feat.shape)
        histogram_of_actions = np.zeros(
            (*quantized_target_feat.shape, self.hoa_bins), dtype=np.uint8
        )
        weights = np.zeros_like(self.ignore_frames[item], dtype=np.float32)
        for i in range(1, self.hoa_window + 1):
            histogram_of_actions[rows[:-i], cols[:-i], quantized_target_feat[:-i]] += 1
            weights[:-i] += 1 - self.ignore_frames[item][i:].astype(np.float32)

        histogram_of_actions = histogram_of_actions / self.hoa_window
        weights = weights / self.hoa_window

        ignore_frames[: -self.hoa_window] = True

        data = dict(
            input=self.input_feats[item],
            target_hist=histogram_of_actions,
            ignore_frames=self.ignore_frames[item],
            ignore_weights=weights,
        )
        return data

    def __len__(self):
        return self.input_feats.shape[0]

    @cached_property
    def input_size(self):
        return self.input_feats.shape[2]

    @cached_property
    def target_size(self):
        return self.target_feats.shape[2]


class KeypointsDataset(Dataset):
    r"""Base class for all datasets.

    Args:
        root (str): Root directory of dataset.
        keypoints (np.ndarray): Array of shape (num_sequences, sequence_len,
            num_feats). Use np.nan for missing values or padding frames.
        annotations (dict(str, np.ndarray)): Dictionary of annotations, where each
            value is an array of shape (num_sequences, sequence_len).
        name (str): Name of the dataset.
        hoa_bins (int): Number of bins for the histograms of actions.
        hoa_window (int): Window size for the histograms of actions.
        force_process (bool): Whether to force process the dataset.
    """

    def __init__(
        self,
        keypoints,
        **kwargs,
    ):
        self.keypoints = keypoints
        input_feats, target_feats, ignore_frames = self.keypoints_to_feats(keypoints)

        super().__init__(input_feats, target_feats, ignore_frames, **kwargs)

    def keypoints_to_feats(self, keypoints):
        # sometimes there are missing frames
        # find frames where any features might be missing
        ignore_frames = np.any(np.isnan(self.keypoints), axis=-1)
        # replace nan values with zeros
        keypoints = np.nan_to_num(self.keypoints)

        # define states and derive actions
        # action[t] = state[t] - state[t-1]
        states = keypoints
        actions = diff(states, axis=1, h=1, padding="edge")

        input_feats = np.stack([states, actions], axis=-1)
        target_feats = actions
        return input_feats, target_feats, ignore_frames
