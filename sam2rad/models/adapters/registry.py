ADAPTER_REGISTRTY = {}


def register_adapter(name):
    """
    Register an adapter.
    """

    def register_adapter_cls(cls):
        if name in ADAPTER_REGISTRTY:
            raise ValueError(f"Cannot register duplicate adapter ({name})")
        ADAPTER_REGISTRTY[name] = cls
        return cls

    return register_adapter_cls
