import numpy as np

import torch
import torch.nn.functional as F
from transformers import (AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer)

from .base_transformer import BaseTransformer
from src.data_classes import ClassificationOutput


class ClassificationTransformer(BaseTransformer):
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

    def predict_sample(self, x, *, max_inp_length=128, return_dict=False):
        """Run network inference and generate predicted output as text."""
        self.model.eval()
        input_ids = self.tokenizer(
            x, return_tensors='pt', max_length=max_inp_length,
            truncation=True, padding=True).input_ids
        input_ids = input_ids.to(self.model.device)

        with torch.no_grad():
            logits = self.model(input_ids).logits
        probs = F.softmax(logits, dim=-1).numpy()

        if return_dict:
            probs = [{self.config.id2label[i]: x for i, x in enumerate(record)}
                     for record in probs]

        return probs

    def get_trainer(self, output_dir, train_dataset=None, eval_dataset=None, *,
                    no_epochs=1, bs=64, gradient_accumulation_steps=1,
                    lr=2e-5, wd=0.01, lr_scheduler_type='linear', fp16=False,
                    compute_metrics_cb=None, num_workers=0, resume_from_checkpoint=None,
                    metric_for_best_model=None, greater_is_better=None,
                    seed=42, log_level='passive', disable_tqdm=False):
        """
        Get the trainer object for training and evaluating the network
        for given training and validation datasets.
        """
        # define training arguments
        args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy='epoch',  # evaluation is done at the end of each epoch
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=no_epochs,
            learning_rate=lr,
            weight_decay=wd,
            lr_scheduler_type=lr_scheduler_type,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            fp16=fp16,  # fp16 is not optimized on the current AWS GPU instance
            dataloader_num_workers=num_workers,
            resume_from_checkpoint=resume_from_checkpoint,
            seed=seed,
            log_level=log_level,
            disable_tqdm=disable_tqdm,
            report_to='none')

        # create data collator for splitting the data into batches
        data_collator = DataCollatorWithPadding(self.tokenizer)

        # create trainer instance
        trainer = Trainer(
            self.model,
            args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics_cb)

        return trainer

    def tokenize_dataset(self, datasets, *, inp_feature='inp', trg_feature='trg',
                         max_inp_length=128):
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
        
        return ClassificationOutput(predictions=preds, label_ids=targs, probs=probs)
