DATASETS = {}


def register_dataset(name):
    """
    Example usage:
    @register_dataset('my_dataset')
    class MyDataset:
        pass
    """

    def register_dataset_cls(cls):
        if name in DATASETS:
            raise ValueError(f"Cannot register duplicate dataset ({name})")
        DATASETS[name] = cls
        return cls

    return register_dataset_cls
