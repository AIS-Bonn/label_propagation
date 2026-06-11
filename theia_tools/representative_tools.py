import json
import numpy as np
from pathlib import Path
import os
from theia_tools.utils import read_image_bbox
from tqdm import tqdm
from random import sample as rsample
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as transforms


class RepresentativeDatasetFO(TorchDataset):
    def __init__(self, fo_dataset, class_labels, label_field="label", transform=None, preprocess=True, repr_num=4) -> None:
        label_distr = {}
        for sample in fo_dataset.iter_samples():
            if sample.ground_truth is None:
                continue
            for detection in sample.ground_truth.detections:
                detects = label_distr.setdefault(detection.label, [])
                detects.append(
                    {
                     "filepath": sample.filepath,
                     "bbox": detection.bounding_box,
                     "label": detection.label
                    }
                )
        self.samples = []
        for key in label_distr.keys():
            if repr_num == None:
                self.samples.extend(label_distr[key])
            else:
                self.samples.extend(rsample(label_distr[key], k=min(repr_num, len(label_distr[key]))))
        self.class_to_ix = {label: i for i, label in enumerate(class_labels)}
        self.ix_to_class = {i: label for i, label in enumerate(class_labels)}
        self.num_classes = len(class_labels)
        self.label_field = label_field
        if preprocess:
            self.pre_processor = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224)])
        else:
            self.pre_processor = None
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dict = self.samples[idx]
        filepath, bbox, label = sample_dict['filepath'], sample_dict['bbox'], sample_dict['label']

        img = read_image_bbox(filepath, bbox)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.pre_processor is not None:
            img = self.pre_processor(img)

        img = torch.tensor(np.array(img))

        return img, label

    def convert_labels(self, labels):
        # Convert list of strings to class indices using your class_to_ix mapping
        indices = torch.tensor([self.class_to_ix[label] for label in labels])

        # Create the target matrix of -1s
        # target_vecs = -1 * torch.ones((len(labels), self.num_classes))

        # Create the target matrix of 0s instead
        target_vecs = torch.zeros((len(labels), self.num_classes))

        # Put +1 at the correct class positions
        target_vecs.scatter_(1, indices.unsqueeze(1), 1.0)

        return target_vecs

    def get_label_names(self, label_ixs):
        labels = [self.ix_to_class[ix] for ix in label_ixs]
        return labels


class RepresentativeMemoryFO:
    def __init__(self, fo_dataset, embedding_model, repr_num=4) -> None:
        label_distr = {}
        print("Choosing representatives from dataset")
        for sample in fo_dataset.iter_samples(progress="verbose"):
            if sample.ground_truth is None:
                continue
            for detection in sample.ground_truth.detections:
                detects = label_distr.setdefault(detection.label, [])
                detects.append(
                    {
                     "filepath": sample.filepath,
                     "bbox": detection.bounding_box,
                     "label": detection.label
                    }
                )
        candidates = []
        for key in label_distr.keys():
            if repr_num == None:
                candidates.extend(label_distr[key])
            else:
                candidates.extend(rsample(label_distr[key], k=min(repr_num, len(label_distr[key]))))
        self.reprs, self.labels = [], []

        device = next(embedding_model.parameters()).device
        for candidate in tqdm(candidates, "Creating embeddings"):
            img = read_image_bbox(candidate["filepath"], candidate["bbox"])
            img = Image.fromarray(img)
            img = torch.tensor(np.array(img), device=device)
            with torch.no_grad():
                self.reprs.append(embedding_model(img))
            self.labels.append(candidate["label"])
        self.reprs = torch.stack(self.reprs, dim=0).squeeze()

    def get_nearest(self, features):
        assert len(features.shape) == 2
        sim = F.cosine_similarity(self.reprs, features)
        score, ix = sim.max(-1)
        score, ix = float(score), int(ix)
        return score, self.labels[ix]

class RepresentativeMemory:
    def __init__(self, annotations_path):
        with open(annotations_path, 'r') as file:
            representatives_data = json.load(file)
        self.representatives_annotations = representatives_data['annotations']
        self.category_map = dict([[str(category_dict['id']), category_dict['name']] for category_dict in representatives_data['categories']])
        self._iterator = iter(self.representatives_annotations)

    def __iter__(self):
        self._iterator = iter(self.representatives_annotations)  # Reset the iterator if needed
        return self

    def __getitem__(self, index):
        return self.representatives_annotations[index]

    def __next__(self):
        annotation = next(self._iterator)
        return annotation


class RepresentativeWriter:
    def __init__(self, annotations_path, images_dir):
        self.memory = RepresentativeMemory(annotations_path)
        self.annotations_path = annotations_path
        with open (annotations_path, 'r') as file:
            data = json.load(file)
        self.image_map = dict([[
            str(image_dict['id']), {
                "file_name": image_dict['file_name'],
                "width": image_dict['width'],
                "height": image_dict['height'],
            }
        ] for image_dict in data['images']])

        self.images_dir = Path(images_dir)
        self.images_dir = self.images_dir if self.images_dir.is_absolute() else Path(os.getcwd()) / self.images_dir
        if not Path.exists(self.images_dir):
            raise FileNotFoundError(str(self.images_dir))
        if not self.images_dir.is_dir():
            raise NotADirectoryError(str(self.images_dir))

    def embed_features(self, model):
        for ann in tqdm(self.memory, desc=f"Generating {model.feature_name} features from representatives"):
            image_id = ann['image_id']
            image_path = self.images_dir / Path(self.image_map[str(image_id)]["file_name"])
            if Path.exists(image_path) and image_path.is_file():
                image = read_image(str(image_path), ann['segmentation'])
                ann[model.feature_name] = model(image).numpy().tolist()
            else:
                raise FileNotFoundError

    def write(self, path=None):
        with open(self.annotations_path, 'r') as file:
            previous_data = json.load(file)
        previous_data["annotations"] = self.memory.representatives_annotations
        if path is None:
            with open(self.annotations_path, "w") as file:
                json.dump(previous_data, file, indent=4)
        else:
            with open(path, "w") as file:
                json.dump(previous_data, file, indent=4)

