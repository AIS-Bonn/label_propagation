import torch
from functools import partial
from torch.utils.data import DataLoader
import torchvision.transforms as T
import fiftyone as fo
import os
import time

from theia_tools.data import load_data_recipe, create_dataset_from_config
from theia_tools.models.clip import CLIPFeatureExtractor
from theia_tools.models.theia import TheiaFeatureExtractor
from theia_tools.models.vit import ViTFeatureExtractor
from theia_tools.fiftyone_bridge import FiftyOneTorchDataset as FOTDataset
from theia_tools.predictors.learners import (
    train_hopfield,
    validate_fo,
    hopfield_ensemble_loss_exp as hopfield_ensemble_loss,
    HopfieldEnsemble as HopfieldEnsembleClassifier,
)
from theia_tools.loggers import MetricLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUNS_DIR = "labeler_models"

# Hyperparams
learning_rate = 1e-3
num_epochs = 20
batch_size = 16
lambda_intra = 0.01
lambda_inter = 0.1

LABELER_CONFIGS = {
    "clip": (
        CLIPFeatureExtractor,
        {"model_type": 'large14', "feature_type": "cls_pool"},
        HopfieldEnsembleClassifier,
        {"feature_dim": None, "proj_dim": 1024, "num_classes": None, "num_heads": 4},
    ),
    "theia": (
        TheiaFeatureExtractor,
        {},
        HopfieldEnsembleClassifier,
        {"feature_dim": None, "proj_dim": 768, "num_classes": None, "num_heads": 4},
    ),
    "ViT": (
        ViTFeatureExtractor,
        {},
        HopfieldEnsembleClassifier,
        {"feature_dim": None, "proj_dim": 1024, "num_classes": None, "num_heads": 4},
    ),
}


def train_val_run(hparams, model_params, run_name, feature_extractor, model, torch_train_dataset, loss_fn):
    lambda_intra, lambda_inter, learning_rate, num_epochs, batch_size = hparams
    train_loader = DataLoader(torch_train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    logger = MetricLogger(
        log_dir=f"{RUNS_DIR}/runs",
        run_name=run_name,
        save_dir=f"{RUNS_DIR}/checkpoints/{run_name}",
    )

    best_loss, best_map = 0., 0.
    for epoch in range(num_epochs):
        print("-" * 10 + f"Epoch {epoch}/{num_epochs-1}" + "-" * 10)
        loss = train_hopfield(model, feature_extractor, train_loader, torch_train_dataset, optimizer, device, loss_fn)
        logger.log_scalar("train/loss", loss, epoch)
        print(f"Training Loss: {loss:.4f}")

        print("Stats on train dataset:")
        fo_results = validate_fo(model, feature_extractor, train_dataset, torch_train_dataset, device)
        logger.log_scalar("train/mAP", fo_results.mAP(), epoch)
        logger.log_confusion_matrix("train/confusion_matrix", fo_results, epoch)
        logger.log_pr_matrix("train/metrics", fo_results, epoch)
        logger.update_best(model, fo_results.mAP(), epoch)

        if fo_results.mAP() > best_map:
            best_loss = loss
            best_map = fo_results.mAP()

    logger.close()
    return {"best_mAP": best_map, "best_loss": best_loss}


if __name__ == "__main__":
    os.makedirs(RUNS_DIR, exist_ok=True)

    for dataset in fo.list_datasets():
        fo.load_dataset(dataset).delete()

    augmentations = T.Compose([
        T.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.2),
        T.RandomAffine(30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        T.ColorJitter(0.3, 0.3, 0.2, 0.05),
        T.RandomGrayscale(0.1),
        T.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)),
        T.CenterCrop(224),
    ])

    data_recipe_train = 'hopfield_train.yaml'
    start = time.time()
    train_dataset, _ = create_dataset_from_config(load_data_recipe(data_recipe_train), verbose=True, name="hopfield_train")
    print(f'Creating train dataset took {time.time()-start:.1f}s.')

    class_labels = train_dataset.distinct("ground_truth.detections.label")
    print(f"Num of labels: {len(class_labels)}")
    torch_train_dataset = FOTDataset(train_dataset, class_labels, transform=augmentations)

    for extractor_name, (extractor_class, extractor_config, model_class, model_config) in LABELER_CONFIGS.items():
        extractor = extractor_class(**extractor_config).to(device)
        model_config = {**model_config, "feature_dim": extractor.feature_dim, "num_classes": len(class_labels)}
        model = model_class(**model_config).to(device)

        hparams = [lambda_intra, lambda_inter, learning_rate, num_epochs, batch_size]
        loss_fn = partial(hopfield_ensemble_loss, lambda_intra=lambda_intra, lambda_inter=lambda_inter)
        print(f"Starting run: {extractor_name}")
        train_val_run(hparams, model_config, extractor_name, extractor, model, torch_train_dataset, loss_fn)
