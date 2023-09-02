from pathlib import Path

import click
import numpy as np
import torchaudio
from torchaudio.functional import resample
from tqdm import tqdm

from acoustic.utils import LogMelSpectrogram

melspectrogram = LogMelSpectrogram()


def process_wav(in_path, out_path):
    wav, sr = torchaudio.load(in_path)
    wav = resample(wav, sr, 16000)

    logmel = melspectrogram(wav.unsqueeze(0))

    np.save(out_path.with_suffix(".npy"), logmel.squeeze().numpy())


def preprocess_dataset(in_dir, out_dir):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting features for {in_dir}")
    for in_path in tqdm(list(in_dir.rglob("*.wav"))):
        relative_path = in_path.relative_to(in_dir)
        out_path = out_dir / relative_path.with_suffix("")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        process_wav(in_path, out_path)


@click.command()
@click.argument("in_dir", type=click.Path(exists=True))
@click.argument("out_dir", type=click.Path())
def main(in_dir, out_dir):
    preprocess_dataset(Path(in_dir), Path(out_dir))


if __name__ == "__main__":
    main()
