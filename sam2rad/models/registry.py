MODEL_REGISTRY = {}


def register_model(name):
    """
    Register a new model.
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Cannot register duplicate model ({name})")
        MODEL_REGISTRY[name] = cls
        return cls

    return register_model_cls
