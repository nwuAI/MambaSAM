MASK_DECODER_REGISTRY = {}

def register_mask_decoder(name):
    def register_mask_decoder_cls(cls):
        if name in MASK_DECODER_REGISTRY:
            raise ValueError(f"Cannot register duplicate mask decoder ({name})")
        MASK_DECODER_REGISTRY[name] = cls
        return cls

    return register_mask_decoder_cls
