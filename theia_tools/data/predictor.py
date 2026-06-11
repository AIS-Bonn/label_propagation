"""
Predictor utilities for applying detection/segmentation models to FiftyOne datasets.
Requires `ultralytics` for YOLO-based models (pip install ultralytics).
"""
import os
import warnings
import yaml
import torch
import cv2
import fiftyone as fo


def apply_predictor(predictor, dataset, verbose=True, label="ground_truth"):
    for sample in dataset.iter_samples(progress=verbose):
        img = cv2.imread(sample.filepath)
        img_h, img_w = img.shape[:2]
        with torch.no_grad():
            results = predictor(img)
        masks = results.get('masks', None)
        if masks is None:
            masks = [None] * len(results['labels'])
        detections = []
        for box_xyxy, conf, lbl, mask in zip(results['boxes_xyxy'], results['conf'], results['labels'], masks):
            x1, y1, x2, y2 = [float(v) for v in box_xyxy]
            bbox = [x1 / img_w, y1 / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h]
            fo_mask = mask.cpu().numpy()[int(y1):int(y2), int(x1):int(x2)] if mask is not None else None
            detections.append(fo.Detection(label=lbl, confidence=float(conf), bounding_box=bbox, mask=fo_mask))
        sample[label] = fo.Detections(detections=detections)
        sample.save()


def load_predictor(model_dir, verbose=False):
    config_path = os.path.join(model_dir, 'export_config.yaml')
    assert os.path.exists(config_path), f"No export_config.yaml found in {model_dir}"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model_type = config['model recipe']['model type']

    if model_type == 'yolo_seg':
        return _YOLOSegPredictor(model_dir, config, verbose)
    else:
        raise NotImplementedError(
            f"Predictor type '{model_type}' is not supported. "
            "Supported types: yolo_seg"
        )


class _YOLOSegPredictor:
    def __init__(self, model_dir, config, verbose=False):
        try:
            from ultralytics import YOLO
            from ultralytics.utils.ops import scale_image
            self._scale_image = scale_image
        except ImportError:
            raise ImportError("ultralytics is required for YOLO predictors: pip install ultralytics")

        engine_path = os.path.join(model_dir, 'model.engine')
        pt_path = os.path.join(model_dir, 'model.pt')
        if os.path.exists(engine_path):
            self.model = YOLO(engine_path, task='segment')
        else:
            warnings.warn('No TensorRT engine found, using PyTorch model.', UserWarning)
            assert os.path.exists(pt_path), f"No model.pt found in {model_dir}"
            self.model = YOLO(pt_path)

        self.model_type = config['model recipe']['model type']
        self.class_labels = config['model recipe']['classes']
        if verbose:
            print(f"Loaded {self.model_type} predictor with {len(self.class_labels)} classes.")

    @torch.no_grad()
    def __call__(self, cv2_image, compute_masks=True):
        out = self.model(cv2_image, verbose=False)[0]
        labels = [self.class_labels[idx.item()] for idx in out.boxes.cls.int().cpu()]
        results = {
            'boxes_xyxy': out.boxes.xyxy,
            'conf': out.boxes.conf,
            'labels': labels,
        }
        if compute_masks and out.masks is not None:
            masks = torch.from_numpy(
                self._scale_image(out.masks.data.permute(1, 2, 0).cpu().numpy(), out.orig_shape)
                .transpose(2, 0, 1)
            ).bool().to(out.boxes.data.device)
            results['masks'] = masks
        return results
