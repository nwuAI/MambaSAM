PROMPT_PREDICTORS = {}


def register_prompt_predictor(name):
    """
    Register prompt predictor.
    """

    def register_model_cls(cls):
        if name in PROMPT_PREDICTORS:
            raise ValueError(f"Cannot register duplicate model ({name})")
        PROMPT_PREDICTORS[name] = cls
        return cls

    return register_model_cls
