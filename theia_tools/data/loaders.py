import os
import yaml
import fiftyone as fo

from theia_tools.data.dataset_utils import (
    select_classes,
    merge_to_superclass,
    concat_datasets,
    fix_crowd_attribute,
)


def load_data_recipe(recipe_name, recipe_dir=None):
    if recipe_dir is None:
        recipe_dir = os.environ.get('RECIPES_DIR', 'recipes')
    with open(os.path.join(recipe_dir, recipe_name), 'r') as f:
        return yaml.safe_load(f)


def _load_dataset_coco_seg(dataset_name, dataset_recipe, verbose=False):
    data_dir = dataset_recipe['data dir']
    labels_dir = dataset_recipe.get('labels dir', data_dir + '/annotations')
    train_dir = dataset_recipe.get('train dir', data_dir + '/train')
    valid_dir = dataset_recipe.get('valid dir', data_dir + '/val')
    train_labels_path = dataset_recipe.get('train labels path', labels_dir + '/instances_Train.json')
    valid_labels_path = dataset_recipe.get('valid labels path', labels_dir + '/instances_Validation.json')

    train_dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=train_dir,
        labels_path=train_labels_path,
        name=dataset_name + '_train',
    )
    valid_dataset = None
    if os.path.exists(valid_dir):
        valid_dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=valid_dir,
            labels_path=valid_labels_path,
            name=dataset_name + '_valid',
        )

    data_field_name = dataset_recipe.get('fiftyone gt field', 'segmentations')
    try:
        train_dataset.delete_sample_field('detections')
        if valid_dataset is not None:
            valid_dataset.delete_sample_field('detections')
    except Exception:
        pass
    train_dataset.rename_sample_field(data_field_name, 'ground_truth')
    if valid_dataset is not None:
        valid_dataset.rename_sample_field(data_field_name, 'ground_truth')

    if verbose:
        print(f'Loaded dataset: {dataset_name}')
        print(train_dataset.distinct("ground_truth.detections.label"))

    return train_dataset, valid_dataset


_DATASET_LOADERS = {
    'coco_seg': _load_dataset_coco_seg,
}


def _load_dataset(dataset_name, dataset_recipe, verbose=False):
    dataset_type = dataset_recipe['dataset type']
    if dataset_type not in _DATASET_LOADERS:
        raise NotImplementedError(f"Dataset type '{dataset_type}' is not supported.")
    train_dataset, valid_dataset = _DATASET_LOADERS[dataset_type](dataset_name, dataset_recipe, verbose=verbose)

    selected_classes = dataset_recipe.get('selected classes', None)
    if selected_classes is not None:
        train_dataset = select_classes(train_dataset, labels=selected_classes)
        if valid_dataset is not None:
            valid_dataset = select_classes(valid_dataset, labels=selected_classes)

    prev_labels = dataset_recipe.get('map classes from', None)
    new_labels = dataset_recipe.get('map classes to', None)
    if prev_labels is not None and new_labels is not None:
        train_dataset = merge_to_superclass(train_dataset, new_labels, prev_labels)
        if valid_dataset is not None:
            valid_dataset = merge_to_superclass(valid_dataset, new_labels, prev_labels)

    if not dataset_recipe.get('validate', True):
        train_datasets = [train_dataset] + ([valid_dataset] if valid_dataset is not None else [])
        return train_datasets, []

    return [train_dataset], ([valid_dataset] if valid_dataset is not None else [])


def create_dataset_from_config(data_config, verbose=False, discard_individual=True, name=None):
    train, valid = [], []
    for dataset_name, dataset_recipe in data_config['datasets'].items():
        train_sets, valid_sets = _load_dataset(dataset_name, dataset_recipe, verbose=verbose)
        train += train_sets
        valid += valid_sets

    train_name = name or 'training'
    valid_name = (name + '_valid') if name else 'validation'
    train_dataset = concat_datasets(train, discard_individual=discard_individual, name=train_name, verbose=verbose)
    valid_dataset = concat_datasets(valid, discard_individual=discard_individual, name=valid_name, verbose=verbose)

    prev_labels = data_config.get('map classes from', None)
    new_labels = data_config.get('map classes to', None)
    if prev_labels is not None and new_labels is not None:
        if train_dataset is not None:
            train_dataset = merge_to_superclass(train_dataset, new_labels, prev_labels)
        if valid_dataset is not None:
            valid_dataset = merge_to_superclass(valid_dataset, new_labels, prev_labels)

    selected_classes = data_config.get('selected classes', None)
    if selected_classes is not None:
        if train_dataset is not None:
            train_dataset = select_classes(train_dataset, labels=selected_classes)
        if valid_dataset is not None:
            valid_dataset = select_classes(valid_dataset, labels=selected_classes)

    if train_dataset is not None:
        train_dataset = fix_crowd_attribute(train_dataset)
    if valid_dataset is not None:
        valid_dataset = fix_crowd_attribute(valid_dataset)

    if verbose:
        print('Finished processing datasets.')
        if train_dataset is not None:
            print('Train labels:', train_dataset.distinct("ground_truth.detections.label"))
        if valid_dataset is not None:
            print('Valid labels:', valid_dataset.distinct("ground_truth.detections.label"))

    return train_dataset, valid_dataset
