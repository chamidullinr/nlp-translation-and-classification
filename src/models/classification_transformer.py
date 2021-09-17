import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification

from .base_transformer import BaseTransformer
from .training_mixin import TrainingMixin
from .model_outputs import ClassificationOutput


__all__ = ['ClassificationTransformer']


class ClassificationTransformer(BaseTransformer, TrainingMixin):
    def __init__(self, pretrained_checkpoint, classes=None):
        super().__init__(pretrained_checkpoint)

        # create params dictionary
        params = dict()
        if classes is not None:
            params['num_labels'] = len(classes)
            params['id2label'] = {i: x for i, x in enumerate(classes)}
            params['label2id'] = {x: i for i, x in enumerate(classes)}

        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_checkpoint, **params)

    def id2label(self, ids):
        return np.array([self.config.id2label[x] for x in ids])

    def label2id(self, labels):
        return np.array([self.config.label2id[x] for x in labels])

    def predict_sample(self, x, *, max_inp_length=None, return_dict=False):
        """Run network inference and generate predicted output as text."""
        self.model.eval()
        model_input = self.tokenizer(
            x, return_tensors='pt', max_length=max_inp_length,
            truncation=True, padding=True)
        model_input = model_input.to(self.model.device)

        with torch.no_grad():
            logits = self.model(**model_input).logits.cpu()
        probs = F.softmax(logits, dim=-1).numpy()

        if return_dict:
            probs = [{self.config.id2label[i]: x for i, x in enumerate(record)}
                     for record in probs]

        return probs

    def tokenize_dataset(self, datasets, *, inp_feature='inp', trg_feature='trg',
                         max_inp_length=None):
        """Tokenize dataset with input and target records before feeding them into the network."""
        def tokenize_records(records):
            inp = ['' if x is None else str(x) for x in records[inp_feature]]
            model_inputs = self.tokenizer(inp, max_length=max_inp_length, truncation=True)
            if trg_feature is not None and trg_feature in records:
                model_inputs['labels'] = [self.config.label2id[x] for x in records[trg_feature]]
            return model_inputs

        return datasets.map(tokenize_records, batched=True)

    def predict(self, test_dataset, *, output_dir='.', bs=64, **kwargs):
        """Apply inference on test dataset, return predictions, labels (optionally) and probs."""
        # create trainer with minimal setup for test-time
        trainer = self.get_trainer(output_dir=output_dir, bs=bs, **kwargs)

        # run inference
        out = trainer.predict(test_dataset)
        logits, targs = out.predictions, out.label_ids
        probs = torch.Tensor(out.predictions).softmax(-1).numpy()
        preds = probs.argmax(-1)

        return ClassificationOutput(predictions=preds, label_ids=targs, probs=probs,
                                    metrics=getattr(out, 'metrics', None))
