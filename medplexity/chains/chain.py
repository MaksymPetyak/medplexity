
class Chain:
    """Chains are used in conjunction with LLM and help to preprocess inputs and outputs for evaluation."""

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("LLM call method is not implemented")
