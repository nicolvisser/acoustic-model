import click
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from acoustic.dataset import MelDataset
from acoustic.module import LitAcousticModel

BATCH_SIZE = 32

dataset_dir = click.prompt(
    "Path to the directory containing the prepared data",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
)

unit_folder_name = click.prompt("Name of the subfolder containing the units", type=str)

version_name = click.prompt("Version name", type=str)

NUM_UNITS = click.prompt("Number of units", type=int)

train_dataset = MelDataset(
    root=dataset_dir,
    num_units=NUM_UNITS,
    train=True,
    unit_folder_name=unit_folder_name,
)

val_dataset = MelDataset(
    root=dataset_dir,
    num_units=NUM_UNITS,
    train=False,
    unit_folder_name=unit_folder_name,
)

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

duration_predictor = LitAcousticModel(num_units=NUM_UNITS, upsample=True)

tensorboard = pl_loggers.TensorBoardLogger(save_dir="", version=version_name)

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
