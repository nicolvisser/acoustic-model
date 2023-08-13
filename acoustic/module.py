import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from acoustic.model import AcousticModel

LEARNING_RATE = 4e-4
BETAS = (0.8, 0.99)
WEIGHT_DECAY = 1e-5


class LitAcousticModel(pl.LightningModule):
    def __init__(
        self,
        num_units: int,
        upsample: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.acoustic_model = AcousticModel(num_units, upsample)

    def training_step(self, batch, batch_idx):
        mels, mels_lengths, units, units_lengths = batch

        mels_ = self.acoustic_model(units, mels[:, :-1, :])

        loss = F.l1_loss(mels_, mels[:, 1:, :], reduction="none")
        loss = torch.sum(loss, dim=(1, 2)) / (mels_.size(-1) * mels_lengths)
        loss = torch.mean(loss)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        mels, units = batch

        mels_ = self.acoustic_model(units, mels[:, :-1, :])

        loss = F.l1_loss(mels_, mels[:, 1:, :])
        self.log("val_loss", loss, prog_bar=True)

        # log image of mel spectrogram
        if batch_idx < 5:  # log only first 5
            for logger in self.trainer.loggers:
                if isinstance(logger, pl.loggers.TensorBoardLogger):
                    img_data = mels_.squeeze().transpose(0, 1)
                    img_data = img_data.unsqueeze(0).cpu().numpy()
                    img_data = img_data - img_data.min()
                    img_data = img_data / (img_data.max() - img_data.min())
                    img_data = (img_data * 255).astype("uint8")
                    img_data = np.flip(img_data, axis=1)
                    logger.experiment.add_image(
                        f"val_mels_{batch_idx}",
                        img_data,
                        self.trainer.global_step,
                    )

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=LEARNING_RATE,
            betas=BETAS,
            weight_decay=WEIGHT_DECAY,
        )
        return optimizer

    def forward(self, x):
        x = self.acoustic_model.encoder(x)
        x = self.acoustic_model.decoder.generate(x)
        return x
