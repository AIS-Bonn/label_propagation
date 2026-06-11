import torch
import fiftyone as fo
import time
import numpy as np
from torch.utils.data import Dataset as TorchDataset, DataLoader
import torchvision.transforms as transforms
from theia_tools.utils import read_image_bbox, read_image_default
from PIL import Image


class FiftyOneDetectionDataset(TorchDataset):
    def __init__(self, fiftyone_dataset, class_labels, label_field="label"):
        self.samples = []
        self.sample_to_detections = {}
        for sample in fiftyone_dataset.iter_samples():
            self.samples.append(sample.filepath)
            for detection in sample.ground_truth.detections:
                if self.sample_to_detections.get(sample.filepath) is None:
                    self.sample_to_detections[sample.filepath] = []
                self.sample_to_detections[sample.filepath].append(detection)
        self.label_field = label_field
        self.class_to_ix = {label: i for i, label in enumerate(class_labels)}
        self.ix_to_class = {i: label for i, label in enumerate(class_labels)}
        self.num_classes = len(class_labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath = self.samples[idx]
        detections = self.sample_to_detections[filepath]
        img = read_image_default(filepath)
        img = torch.tensor(img)
        detections = [{"bbox": d.bounding_box, "label": d.label, "mask": d.mask} for d in detections]
        return img, detections

    def convert_labels(self, labels):
        indices = torch.tensor([self.class_to_ix[label] for label in labels])
        target_vecs = torch.zeros((len(labels), self.num_classes))
        target_vecs.scatter_(1, indices.unsqueeze(1), 1.0)
        return target_vecs

    def get_label_names(self, label_ixs):
        return [self.ix_to_class[int(ix)] for ix in label_ixs]


class FiftyOneTorchDataset(TorchDataset):
    def __init__(self, fiftyone_dataset, class_labels, label_field="label", transform=None, preprocess=True):
        self.samples = []
        for sample in fiftyone_dataset.iter_samples():
            if sample.ground_truth is None:
                continue
            for detection in sample.ground_truth.detections:
                self.samples.append((sample.filepath, detection))
        self.pre_processor = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224)]) if preprocess else None
        self.label_field = label_field
        self.class_to_ix = {label: i for i, label in enumerate(class_labels)}
        self.ix_to_class = {i: label for i, label in enumerate(class_labels)}
        self.num_classes = len(class_labels)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, detection = self.samples[idx]
        img = read_image_bbox(filepath, detection.bounding_box)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.pre_processor is not None:
            img = self.pre_processor(img)
        img = torch.tensor(np.array(img))
        label = detection[self.label_field]
        return img, label

    def convert_labels(self, labels):
        indices = torch.tensor([self.class_to_ix[label] for label in labels])
        target_vecs = torch.zeros((len(labels), self.num_classes))
        target_vecs.scatter_(1, indices.unsqueeze(1), 1.0)
        return target_vecs

    def get_label_names(self, label_ixs):
        return [self.ix_to_class[ix] for ix in label_ixs]


class TorchSegmentationsDataset(TorchDataset):
    def __init__(self, fiftyone_dataset, image_read_fn, label_to_ix_mapping):
        self.samples = []
        self.label_to_ix = label_to_ix_mapping
        for sample in fiftyone_dataset.iter_samples(progress=True):
            self.samples.append((sample.id, image_read_fn(sample.filepath), self._segmentation_from_detections(sample)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fo_id, img, segmentation_gt = self.samples[idx]
        img = torch.tensor(np.array(img))
        return fo_id, img, segmentation_gt

    def _segmentation_from_detections(self, fo_sample):
        H, W = fo_sample.metadata.height, fo_sample.metadata.width
        seg = torch.zeros((H, W), dtype=int, requires_grad=False)
        for det in fo_sample.ground_truth.detections:
            cls_idx = self.label_to_ix[det.label]
            x_rel, y_rel, w_rel, h_rel = det.bounding_box
            left = max(0, min(int(round(x_rel * W)), W - 1))
            top = max(0, min(int(round(y_rel * H)), H - 1))
            right = max(0, min(left + int(round(w_rel * W)), W))
            bottom = max(0, min(top + int(round(h_rel * H)), H))
            seg[top:bottom, left:right][det.mask] = cls_idx
        return seg


class Dataset:
    def __init__(self, dataset_config, validation_config, first_dataset=False):
        from theia_tools.data import load_data_recipe, create_dataset_from_config

        self.dataset_name = dataset_config["dataset_name"]
        if first_dataset:
            for dataset in fo.list_datasets():
                fo.load_dataset(dataset).delete()

        start = time.time()
        self.fo_dataset, _ = create_dataset_from_config(
            load_data_recipe(dataset_config["recipe_name"]),
            verbose=True,
            name=self.dataset_name,
        )
        print(f'Creating (FiftyOne) {self.dataset_name} dataset took {time.time()-start:.1f}s.')

        class_labels = ["background"] + self.fo_dataset.distinct("ground_truth.detections.label")
        self.class_to_ix = {label: i for i, label in enumerate(class_labels)}
        self.ix_to_class = {i: label for i, label in enumerate(class_labels)}
        self.num_classes = len(class_labels)

        start = time.time()
        self.torch_dataset = TorchSegmentationsDataset(self.fo_dataset, read_image_default, self.class_to_ix)
        self.torch_loader = DataLoader(
            self.torch_dataset,
            batch_size=dataset_config['batch_size'],
            shuffle=dataset_config["shuffle"],
            num_workers=dataset_config['num_workers'],
        )
        print(f'Creating (Torch) {self.dataset_name} dataset took {time.time()-start:.1f}s.')

        if "segmentations" in validation_config["validations"]:
            self._detections_to_segmentations()

    def _detections_to_segmentations(self):
        for sample in self.fo_dataset:
            segmentation = self.torch_dataset._segmentation_from_detections(sample)
            sample["ground_truth_seg"] = fo.Segmentation(mask=np.array(segmentation))
            sample.save()

    def __iter__(self):
        return iter(self.torch_loader)

    def __len__(self):
        return len(self.torch_loader)

    def __getitem__(self, sample_id):
        return self.fo_dataset[sample_id]

    def samples_fo(self):
        for batch in self.torch_loader:
            fo_ids, imgs, _ = batch
            yield fo_ids, imgs

    def evaluate_detections(self, *args, **kwargs):
        return self.fo_dataset.evaluate_detections(*args, **kwargs)

    def evaluate_segmentations(self, *args, **kwargs):
        return self.fo_dataset.evaluate_segmentations(*args, **kwargs)
