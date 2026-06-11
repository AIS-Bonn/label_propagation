import torch
import numpy as np
import fiftyone as fo
import os
import time
from PIL import Image

from theia_tools.data import load_data_recipe, create_dataset_from_config, color_by_instance
from theia_tools.models.clip import CLIPFeatureExtractor
from theia_tools.models.theia import TheiaFeatureExtractor
from theia_tools.models.vit import ViTFeatureExtractor
from theia_tools.utils import read_image_bbox
from theia_tools.fiftyone_bridge import FiftyOneTorchDataset as FOTDataset
from theia_tools.predictors.learners import HopfieldEnsemble as HopfieldEnsembleClassifier, HopfieldBoost

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELERS_DIR = "labeler_models/checkpoints"
REPR_DATASET_RECIPE = "hopfield_train.yaml"
VALID_DATASET_RECIPE = "hopfield_valid.yaml"

if __name__ == "__main__":
    for dataset in fo.list_datasets():
        fo.load_dataset(dataset).delete()

    if not os.path.exists(LABELERS_DIR):
        raise FileNotFoundError(f"Labelers directory not found: {LABELERS_DIR}")

    start = time.time()
    repr_dataset, _ = create_dataset_from_config(load_data_recipe(REPR_DATASET_RECIPE), verbose=True, name="representatives")
    class_labels = repr_dataset.distinct("ground_truth.detections.label")
    print(f'Creating representatives dataset took {time.time()-start:.1f}s.')

    start = time.time()
    valid_dataset, _ = create_dataset_from_config(load_data_recipe(VALID_DATASET_RECIPE), verbose=True, name="validations")
    print(f'Creating validations dataset took {time.time()-start:.1f}s.')

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

    torch_dataset = FOTDataset(valid_dataset, class_labels)
    print(f"Labeling with {boost_model.feature_name}")

    for sample in valid_dataset.iter_samples(progress="verbose"):
        detections = []
        for gt_detection in sample.ground_truth.detections:
            img = read_image_bbox(sample.filepath, gt_detection.bounding_box)
            img = torch_dataset.pre_processor(Image.fromarray(img))
            img = torch.tensor(np.array(img)).unsqueeze(0)

            features_list = []
            with torch.no_grad():
                for extractor in extractors:
                    features_list.append(extractor(img))
                pred_ix, pred_attn = boost_model.predict(features_list)

            pred_label = torch_dataset.get_label_names([int(pred_ix)])[0]
            detections.append(fo.Detection(
                label=pred_label,
                confidence=float(pred_attn),
                bounding_box=gt_detection.bounding_box,
                mask=gt_detection.mask,
            ))
        sample[boost_model.feature_name] = fo.Detections(detections=detections)
        sample.save()

    print("-" * 100)
    eval_result = valid_dataset.evaluate_detections(
        boost_model.feature_name,
        gt_field="ground_truth",
        use_masks=True,
        compute_mAP=True,
    )
    eval_result.print_report()
    print(f"mAP score {eval_result.mAP():.3f}")
    print("-" * 100)

    session = fo.launch_app(valid_dataset, remote=True)
    color_by_instance(valid_dataset, session)
    input("Done!")
