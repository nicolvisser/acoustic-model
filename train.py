from pathlib import Path

import click
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from acoustic.dataset import MelDataset
from acoustic.module import LitAcousticModel

BATCH_SIZE = 32


@click.command()
@click.option(
    "--mels_dir",
    type=click.Path(exists=True, file_okay=False),
    help="Path to the directory containing the mel spectrograms",
    prompt=True,
)
@click.option(
    "--units_dir",
    type=click.Path(exists=True, file_okay=False),
    help="Path to the directory containing the units",
    prompt=True,
)
@click.option(
    "--num_units",
    type=int,
    help="Number of units",
    prompt=True,
)
@click.option(
    "--version_name",
    type=str,
    help="Checkpoint version name",
    prompt=True,
)
def train(mels_dir: str, units_dir: str, num_units: int, version_name: str):
    train_dataset = MelDataset(mels_dir=mels_dir, units_dir=units_dir, num_units=num_units, train=True)

    val_dataset = MelDataset(
        mels_dir=mels_dir,
        units_dir=units_dir,
        num_units=num_units,
        train=False,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=train_dataset.pad_collate,
        pin_memory=True,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    duration_predictor = LitAcousticModel(num_units=num_units, upsample=True)

    tensorboard = pl_loggers.TensorBoardLogger(save_dir="", version=version_name)

    if Path(tensorboard.log_dir).exists():
        raise ValueError(
            f"A log with version name, {version_name}, already exists in {tensorboard.log_dir}. Please choose another version name or delete the existing version."
        )

    checkpoint_callback = ModelCheckpoint(save_top_k=3, save_last=True, monitor="val_loss")

    trainer = pl.Trainer(
        accelerator="gpu",
        logger=tensorboard,
        max_epochs=65,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(
        model=duration_predictor,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    train()
