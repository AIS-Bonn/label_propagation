import time
import os
import argparse
import torch
import numpy as np
import fiftyone as fo
from fiftyone import ViewField as F
from PIL import Image

from theia_tools.data import (
    load_data_recipe,
    create_dataset_from_config,
    load_predictor,
    apply_predictor,
    fix_crowd_attribute,
    color_by_instance,
)
from theia_tools.models.theia import TheiaFeatureExtractor
from theia_tools.models.vit import ViTFeatureExtractor
from theia_tools.models.clip import CLIPFeatureExtractor
from theia_tools.utils import read_image_bbox
from theia_tools.fiftyone_bridge import FiftyOneTorchDataset as FOTDataset
from theia_tools.predictors.learners import HopfieldEnsemble as HopfieldEnsembleClassifier, HopfieldBoost

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--to_label_dir', default="datasets")
parser.add_argument('--detector_dir', default="/mounted_models/Singular/singular_object_test")
parser.add_argument('--labelers_dir', default="labeler_models/checkpoints")
parser.add_argument('--repr_dataset_recipe', default="hopfield_train.yaml")
parser.add_argument('--export_dir', default="export_test")
parser.add_argument('--label_objects', action='store_true', default=False)
args = parser.parse_args()

ROOT = args.to_label_dir
MODEL_DIR = args.detector_dir
LABELERS_DIR = args.labelers_dir
REPR_DATASET_RECIPE = args.repr_dataset_recipe
EXPORT_DIR = args.export_dir
LABEL_OBJECTS = args.label_objects

for dataset in fo.list_datasets():
    fo.load_dataset(dataset).delete()

if __name__ == "__main__":
    start = time.time()
    repr_dataset, _ = create_dataset_from_config(load_data_recipe(REPR_DATASET_RECIPE), verbose=True, name="representatives")
    class_labels = repr_dataset.distinct("ground_truth.detections.label")
    print(f'Creating representatives dataset took {time.time()-start:.1f}s.')

    ds_list = []
    for ds_name in sorted(os.listdir(ROOT)):
        ds_path = os.path.join(ROOT, ds_name)
        if not os.path.isdir(ds_path):
            continue
        print(f"\n--- Processing dataset '{ds_name}' ---")
        dataset = fo.Dataset(ds_name, overwrite=True)
        for split in ("train", "valid"):
            split_dir = os.path.join(ds_path, split)
            for imgfile in os.listdir(split_dir):
                sample = fo.Sample(filepath=os.path.join(split_dir, imgfile))
                sample["split"] = split
                dataset.add_sample(sample)
        ds_list.append(dataset)
        print(f"Added dataset '{ds_name}'.")

    predictor = load_predictor(MODEL_DIR, verbose=True)
    for dataset in ds_list:
        apply_predictor(predictor, dataset, verbose=True, label="ground_truth")
    del predictor

    if LABEL_OBJECTS:
        theia_extractor = TheiaFeatureExtractor().to(device)
        clip_extractor = CLIPFeatureExtractor('large14', 'cls_pool').to(device)
        vit_extractor = ViTFeatureExtractor().to(device)

        clip_classifier = HopfieldEnsembleClassifier.load_from_checkpoint(
            f"{LABELERS_DIR}/clip/best_model.pth", feature_dim=1024, proj_dim=1024, num_classes=len(class_labels), num_heads=4
        ).to(device)
        theia_classifier = HopfieldEnsembleClassifier.load_from_checkpoint(
            f"{LABELERS_DIR}/theia/best_model.pth", feature_dim=768, proj_dim=768, num_classes=len(class_labels), num_heads=4
        ).to(device)
        vit_classifier = HopfieldEnsembleClassifier.load_from_checkpoint(
            f"{LABELERS_DIR}/ViT/best_model.pth", feature_dim=1280, proj_dim=1024, num_classes=len(class_labels), num_heads=4
        ).to(device)

        extractors = (clip_extractor, theia_extractor, vit_extractor)
        boost_model = HopfieldBoost((clip_classifier, theia_classifier, vit_classifier), learn_weights=False).to(device)
        boost_model.eval()

        for dataset in ds_list:
            torch_dataset = FOTDataset(dataset, class_labels)
            for sample in dataset.iter_samples(progress="verbose"):
                for gt_detection in sample.ground_truth.detections:
                    img = read_image_bbox(sample.filepath, gt_detection.bounding_box)
                    img = torch_dataset.pre_processor(Image.fromarray(img))
                    img = torch.tensor(np.array(img)).unsqueeze(0)
                    features_list = []
                    with torch.no_grad():
                        for extractor in extractors:
                            features_list.append(extractor(img))
                        pred_ix, _ = boost_model.predict(features_list)
                    gt_detection.label = torch_dataset.get_label_names([int(pred_ix)])[0]
                sample.save()

    print("Exporting labeled datasets...")
    for dataset in ds_list:
        session = fo.launch_app(dataset, remote=True)
        fix_crowd_attribute(dataset, value=1)
        color_by_instance(dataset, session)

        os.makedirs(os.path.join(EXPORT_DIR, dataset.name), exist_ok=True)
        for split in ("train", "valid"):
            label_file = "instances_Train.json" if split == "train" else "instances_Validation.json"
            dataset.match(F("split") == split).export(
                export_dir=os.path.join(EXPORT_DIR, dataset.name, split),
                dataset_type=fo.types.COCODetectionDataset,
                label_field="ground_truth",
                include_masks=True,
                export_media=True,
                labels_path=label_file,
            )
        print(f"Exported to {EXPORT_DIR}!")

    input("Done!")
