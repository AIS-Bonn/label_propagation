import fiftyone as fo
from fiftyone import ViewField as F


def fix_crowd_attribute(dataset, fields=None, value=0):
    if fields is None:
        fields = ['ground_truth']
    if dataset is None:
        return dataset
    for sample in dataset.iter_samples(progress=True):
        for field in fields:
            if field in sample and sample[field] is not None:
                for detection in sample[field].detections:
                    detection.iscrowd = value
        sample.save()
    return dataset


def select_classes(dataset, labels=None, mode='include'):
    if labels is None:
        return dataset
    if isinstance(labels, str):
        labels = [labels]
    if mode == 'include':
        dataset = dataset.filter_labels("ground_truth", F("label").is_in(labels))
    elif mode == 'exclude':
        dataset = dataset.filter_labels("ground_truth", ~F("label").is_in(labels))
    else:
        raise NotImplementedError(f"Mode '{mode}' is not supported.")
    return dataset


def merge_to_superclass(dataset, superclasses, labels):
    if superclasses is None:
        return dataset
    if isinstance(superclasses, str):
        superclasses = [superclasses]
    if isinstance(labels, str):
        labels = [labels]
    else:
        labels = [[label] if isinstance(label, str) else label for label in labels]
    assert len(superclasses) == len(labels)
    for superclass, class_labels in zip(superclasses, labels):
        mapping = {label: superclass for label in class_labels}
        dataset = dataset.map_labels("ground_truth", mapping)
    return dataset


def concat_datasets(datasets, discard_individual=True, name=None, verbose=False):
    if not datasets:
        return None
    dataset_ret = fo.Dataset(name)
    if verbose:
        print(f'Merging datasets into "{name}"')
    for dataset in datasets:
        for sample in dataset.iter_samples(progress=verbose):
            dataset_ret.add_sample(sample)
    if discard_individual:
        for dataset in datasets:
            if hasattr(dataset, 'dataset_name'):
                fo.load_dataset(dataset.dataset_name).delete()
            else:
                dataset.delete()
    return dataset_ret


def color_by_instance(dataset, session):
    color_scheme = session.color_scheme
    color_scheme.color_by = 'instance'
    dataset.app_config.color_scheme = color_scheme
    dataset.save()
    session.refresh()
