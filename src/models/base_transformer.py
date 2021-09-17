from transformers import AutoTokenizer


__all__ = ['BaseTransformer']


class BaseTransformer:
    def __init__(self, pretrained_checkpoint):
        self.pretrained_checkpoint = pretrained_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint)
        self.model = None

    def __str__(self):
        return f'{self.__class__.__name__}({self.pretrained_checkpoint})'

    def __repr__(self):
        return str(self)

    def from_pretrained(self, filename):
        self.model = self.model.from_pretrained(filename)
        return self

    def save_pretrained(self, filename):
        self.model.save_pretrained(filename)
        return self

    def num_trainable_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @property
    def config(self):
        return self.model.config if self.model is not None else None

    def predict_sample(self, *args, **kwargs):
        raise NotImplementedError

    def tokenize_dataset(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError
