dependencies = ["torch"]

URLS = {
    "hubert-bshall": {
        "ljspeech": {
            50: {
                0: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-50-ce7b91ce.ckpt"
            },
            100: {
                0: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-100-ac6ec4d5.ckpt",
                4: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-100-dp-lambda-4-0041575b.ckpt",
                8: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-100-dp-lambda-8-bdb4d455.ckpt",
                12: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-100-dp-lambda-12-487cf2d7.ckpt",
                16: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-100-dp-lambda-16-1db07a1f.ckpt",
                20: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-100-dp-lambda-20-cd0ef321.ckpt",
            },
            200: {
                0: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-200-fb45eba7.ckpt",
                4: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-200-dp-lambda-4-82e6c200.ckpt",
                8: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-200-dp-lambda-8-2fbdd6da.ckpt",
                12: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-200-dp-lambda-12-aeaaaa34.ckpt",
                16: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-200-dp-lambda-16-81d50a02.ckpt",
                20: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-200-dp-lambda-20-b5d019d4.ckpt",
            },
            500: {
                0: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-500-239bee21.ckpt",
                4: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-500-dp-lambda-4-5850c2ff.ckpt",
                8: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-500-dp-lambda-8-35589d7d.ckpt",
                12: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-500-dp-lambda-12-82f7a0e8.ckpt",
                16: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-500-dp-lambda-16-69a7ce5e.ckpt",
                20: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-500-dp-lambda-20-a602c78f.ckpt",
                24: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-500-dp-lambda-24-c3808155.ckpt",
                28: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-500-dp-lambda-28-27435c00.ckpt",
            },
            1000: {
                0: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-1000-6d5f80f7.ckpt",
                4: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-1000-dp-lambda-4-4ab70eb5.ckpt",
                8: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-1000-dp-lambda-8-80d17ebb.ckpt",
                12: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-1000-dp-lambda-12-98958967.ckpt",
                16: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-1000-dp-lambda-16-e44c90b1.ckpt",
                20: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-1000-dp-lambda-20-e6a6a307.ckpt",
            },
            2000: {
                0: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-2000-586b792b.ckpt",
                4: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-2000-dp-lambda-4-7b63192c.ckpt",
                8: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-2000-dp-lambda-8-0a475d6c.ckpt",
                12: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-2000-dp-lambda-12-c9884837.ckpt",
                16: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-2000-dp-lambda-16-39c37f2a.ckpt",
                20: "https://github.com/nicolvisser/acoustic-model/releases/download/v0.1/acoustic-hubert-bshall-ljspeech-kmeans-2000-dp-lambda-20-f411f1cb.ckpt",
            },
        },
    }
}

import torch

from acoustic import AcousticModel


def acoustic(
    features: str = "hubert-bshall",
    dataset: str = "ljspeech",
    n_units: int = 500,
    dp_lmbda: int = 0,
    pretrained: bool = True,
    progress: bool = True,
) -> AcousticModel:
    # Check that the dataset, n_clusters and dp_smoothing_lambda are available

    allowed_features = URLS.keys()
    assert features in allowed_features, f"features must be one of {allowed_features}"
    allowed_datasets = URLS[features].keys()
    assert dataset in allowed_datasets, f"dataset must be one of {allowed_datasets}, if you choose {features}"
    allowed_n_clusters = URLS[features][dataset].keys()
    assert (
        n_units in allowed_n_clusters
    ), f"n_clusters must be one of {allowed_n_clusters}, if you choose {features} and {dataset}"
    allowed_lmbdas = URLS[features][dataset][n_units].keys()
    assert (
        dp_lmbda in allowed_lmbdas
    ), f"dp_smoothing_lambda must be one of {allowed_lmbdas}, if you choose {features}, {dataset} and {n_units} units"

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            URLS[features][dataset][n_units][dp_lmbda],
            progress=progress,
            check_hash=True,
        )
        model = AcousticModel.load_model_from_lit_checkpoint(checkpoint)
        model.eval()
    else:
        model = AcousticModel(num_units=n_units, upsample=True)

    return model
