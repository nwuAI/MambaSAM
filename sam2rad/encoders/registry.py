IMAGE_ENCODER_REGISTRY = {}


def register_image_encoder(name):
    """
    Register a new image encoder.
    """

    def register_image_encoder_cls(cls):
        if name in IMAGE_ENCODER_REGISTRY:
            raise ValueError(f"Cannot register duplicate image encoder ({name})")
        IMAGE_ENCODER_REGISTRY[name] = cls
        return cls

    return register_image_encoder_cls
