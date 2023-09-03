dependencies = ["torch"]

URLS = {
    "ljspeech": {
        50: {
            0: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/ljspeech-kmeans-50-ce7b91ce.ckpt"
        },
        100: {
            0: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/ljspeech-kmeans-100-ac6ec4d5.ckpt"
        },
        200: {
            0: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/ljspeech-kmeans-200-fb45eba7.ckpt"
        },
        500: {
            0: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/ljspeech-kmeans-500-239bee21.ckpt",
            4: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/ljspeech-kmeans-500-dp-lambda-4-5850c2ff.ckpt",
            8: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/ljspeech-kmeans-500-dp-lambda-8-35589d7d.ckpt",
            12: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/ljspeech-kmeans-500-dp-lambda-12-82f7a0e8.ckpt",
            16: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/ljspeech-kmeans-500-dp-lambda-16-69a7ce5e.ckpt",
        },
        1000: {
            0: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/ljspeech-kmeans-1000-6d5f80f7.ckpt",
            4: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/ljspeech-kmeans-1000-dp-lambda-4-4ab70eb5.ckpt",
        },
        2000: {
            0: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/ljspeech-kmeans-2000-586b792b.ckpt"
        },
    },
}

import torch

from acoustic import AcousticModel
from load_from_checkpoint import _load_model_from_checkpoint


def acoustic(
    dataset: str,
    n_clusters: int,
    lmbda: int = 0,
    pretrained: bool = True,
    progress: bool = True,
) -> AcousticModel:
    # Check that the dataset, n_clusters and dp_smoothing_lambda are available
    allowed_datasets = URLS.keys()
    assert dataset in allowed_datasets, f"dataset must be one of {allowed_datasets}"
    allowed_n_clusters = URLS[dataset].keys()
    assert (
        n_clusters in allowed_n_clusters
    ), f"n_clusters must be one of {allowed_n_clusters} when using {dataset} dataset"

    if lmbda > 0:
        allowed_lmbdas = URLS[dataset][n_clusters].keys()
        assert (
            lmbda in allowed_lmbdas
        ), f"lmbda must be one of {allowed_lmbdas} when using {dataset} dataset and {n_clusters} clusters"

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            URLS[dataset][n_clusters][lmbda],
            progress=progress,
            check_hash=True,
        )
        model = _load_model_from_checkpoint(checkpoint)
        model.eval()
    else:
        model = AcousticModel(num_units=n_clusters, upsample=True)

    return model
