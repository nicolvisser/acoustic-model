from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class MelDataset(Dataset):
    def __init__(self, mels_dir: Path, units_dir: Path, num_units: int, train: bool = True):
        self.mels_dir = Path(mels_dir)
        self.units_dir = Path(units_dir)

        assert self.mels_dir.exists(), f"{self.mels_dir} does not exist"
        assert self.units_dir.exists(), f"{self.units_dir} does not exist"

        self.num_units = num_units

        pattern = "train/**/*.npy" if train else "dev/**/*.npy"
        self.metadata = [
            path.relative_to(self.mels_dir).with_suffix("")
            for path in self.mels_dir.rglob(pattern)
        ]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        path = self.metadata[index]
        mel_path = self.mels_dir / path
        units_path = self.units_dir / path

        mel = np.load(mel_path.with_suffix(".npy")).T
        units = np.load(units_path.with_suffix(".npy"))

        length = 2 * units.shape[0]

        mel = torch.from_numpy(mel[:length, :])
        mel = F.pad(mel, (0, 0, 1, 0))
        units = torch.from_numpy(units).long()

        return mel, units

    def pad_collate(self, batch):
        mels, units = zip(*batch)

        mels, units = list(mels), list(units)

        mels_lengths = torch.tensor([x.size(0) - 1 for x in mels])
        units_lengths = torch.tensor([x.size(0) for x in units])

        mels = pad_sequence(mels, batch_first=True)
        units = pad_sequence(
            units, batch_first=True, padding_value=self.num_units
        )  # index self.num_units is the padding index

        return mels, mels_lengths, units, units_lengths
