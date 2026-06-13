"""Teacher dataset package."""

from importlib import import_module

_EXPORTS = {
    "GTAUAVDataset": ("src.dataset.teacher.val_dataloaders", "GTAUAVDataset"),
    "IdentityBatchSampler": ("src.dataset.teacher.datasets", "IdentityBatchSampler"),
    "IdentityU1652Dataset": ("src.dataset.teacher.datasets", "IdentityU1652Dataset"),
    "IndexedDataset": ("src.dataset.teacher.val_dataloaders", "IndexedDataset"),
    "Sample4GeoBatchSampler": ("src.dataset.teacher.datasets", "Sample4GeoBatchSampler"),
    "Sample4GeoU1652Dataset": ("src.dataset.teacher.datasets", "Sample4GeoU1652Dataset"),
    "Sample4GeoU1652DatasetEval": (
        "src.dataset.teacher.val_dataloaders",
        "Sample4GeoU1652DatasetEval",
    ),
    "build_1652_val_dataloaders": (
        "src.dataset.teacher.val_dataloaders",
        "build_1652_val_dataloaders",
    ),
    "build_gta_val_dataloaders": (
        "src.dataset.teacher.val_dataloaders",
        "build_gta_val_dataloaders",
    ),
    "build_sues200_val_dataloaders": (
        "src.dataset.teacher.val_dataloaders",
        "build_sues200_val_dataloaders",
    ),
    "collate_identity_u1652_batch": (
        "src.dataset.teacher.datasets",
        "collate_identity_u1652_batch",
    ),
    "create_1652_teacher_train_dataloaders": (
        "src.dataset.teacher.datasets",
        "create_1652_teacher_train_dataloaders",
    ),
    "create_1652_train_dataset": ("src.dataset.teacher.datasets", "create_1652_train_dataset"),
    "create_identity_1652_train_dataset": (
        "src.dataset.teacher.datasets",
        "create_identity_1652_train_dataset",
    ),
    "get_sample4geo_train_transforms": (
        "src.dataset.teacher.transforms",
        "get_sample4geo_train_transforms",
    ),
    "get_sample4geo_val_transforms": (
        "src.dataset.teacher.transforms",
        "get_sample4geo_val_transforms",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
