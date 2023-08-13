from pathlib import Path

import click
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from acoustic.dataset import MelDataset
from acoustic.module import LitAcousticModel

NUM_UNITS = 100
BATCH_SIZE = 32

dataset_dir = Path(
    click.prompt(
        "Path to the directory containing the prepared data",
        type=click.Path(exists=True, dir_okay=True, file_okay=False),
    )
)

train_dataset = MelDataset(root=dataset_dir, num_units=NUM_UNITS, train=True)

val_dataset = MelDataset(root=dataset_dir, num_units=NUM_UNITS, train=False)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    collate_fn=train_dataset.pad_collate,
    pin_memory=True,
    drop_last=True,
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    drop_last=False,
)

duration_predictor = LitAcousticModel(num_units=100, upsample=True)

tensorboard = pl_loggers.TensorBoardLogger(save_dir="")

checkpoint_callback = ModelCheckpoint(save_top_k=3, save_last=True, monitor="val_loss")

trainer = pl.Trainer(
    accelerator="gpu",
    logger=tensorboard,
    max_epochs=-1,
    log_every_n_steps=10,
    callbacks=[checkpoint_callback],
)

trainer.fit(
    model=duration_predictor,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
