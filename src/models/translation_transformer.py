import torch
import torch.nn.functional as F
from transformers import (AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer)
from transformers.trainer import logger
from transformers.trainer_pt_utils import torch_pad_and_concatenate as concat

from src.torch_utils import torch_isin
from src.data_classes import TranslationOutput
from .base_transformer import BaseTransformer


class TranslationTransformer(BaseTransformer):
    SKIP_PROB_TOK = -1

    def __init__(self, pretrained_checkpoint):
        super().__init__(pretrained_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_checkpoint)

    def _generate(self, input_ids, output_probs, **kwargs):
        self.model.eval()
        if output_probs:
            with torch.no_grad():
                out_dict = self.model.generate(
                    input_ids, output_scores=True, return_dict_in_generate=True, **kwargs)
            logits = out_dict['sequences'].cpu()
            scores = torch.cat([x.cpu().unsqueeze(1) for x in out_dict['scores']], 1)
            probs, _ = F.softmax(scores, -1).max(-1)
            # set prob of special tokens to -1
            probs[torch_isin(logits[:, 1:], self.tokenizer.all_special_ids)] = self.SKIP_PROB_TOK
        else:
            with torch.no_grad():
                logits = self.model.generate(input_ids, **kwargs).cpu()
            probs = None
        return logits, probs

    def predict_sample(self, x, output_probs=False, *,
                       max_inp_length=128, max_trg_length=128, **kwargs):
        """Run network inference and generate predicted output as text."""
        input_ids = self.tokenizer(
            x, return_tensors='pt', max_length=max_inp_length,
            truncation=True, padding=True).input_ids
        input_ids = input_ids.to(self.model.device)
        logits, probs = self._generate(
            input_ids, output_probs, max_length=max_trg_length, **kwargs)
        decoded = self.tokenizer.batch_decode(logits, skip_special_tokens=True)
        if len(decoded) == 1:
            decoded = decoded[0]
        out = decoded if probs is None else (decoded, probs)
        return out

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
        args = Seq2SeqTrainingArguments(
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
            predict_with_generate=True,
            fp16=fp16,  # fp16 is not optimized on the current AWS GPU instance
            dataloader_num_workers=num_workers,
            resume_from_checkpoint=resume_from_checkpoint,
            seed=seed,
            log_level=log_level,
            disable_tqdm=disable_tqdm,
            report_to='none')

        # create data collator for splitting the data into batches
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        # create trainer instance
        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics_cb)

        return trainer

    def tokenize_dataset(self, datasets, *, inp_feature='inp', trg_feature='trg',
                         max_inp_length=128, max_trg_length=128, prefix=None, prefix_col=None):
        """Tokenize dataset with input and target records before feeding them into the network."""
        def tokenize_records(records):
            inp = ['' if x is None else str(x) for x in records[inp_feature]]
            if prefix is not None:
                inp = [f'{pref} {x}' for x in inp]
            elif prefix_col is not None and prefix_col in records:
                inp = [f'{pref} {x}' for x, pref in zip(inp, records[prefix_col])]
            model_inputs = self.tokenizer(inp, max_length=max_inp_length, truncation=True)
            if trg_feature is not None and trg_feature in records:
                trg = ['' if x is None else str(x) for x in records[trg_feature]]
                with self.tokenizer.as_target_tokenizer():
                    model_inputs['labels'] = self.tokenizer(
                        trg, max_length=max_trg_length, truncation=True).input_ids
            else:
                # due to a shortcoming in transformers==4.8.2 where 'labels' cannot be empty
                model_inputs['labels'] = model_inputs.input_ids
            return model_inputs

        tokenized_datasets = datasets.map(tokenize_records, batched=True)
        return tokenized_datasets

    def predict(self, test_dataset, *, output_dir='.', bs=64,
                output_probs=False, max_length=None, **kwargs):
        """Apply inference on test dataset, return predictions, labels (optionally) and probs (optionally)."""
        # create trainer with minimal setup for test-time
        trainer = self.get_trainer(output_dir=output_dir, bs=bs, **kwargs)

        # create PyTorch dataloader
        dataloader = trainer.get_test_dataloader(test_dataset)

        # get model
        model = trainer._wrap_model(trainer.model, training=False)
        max_length = max_length or model.config.max_length

        # set mixed precision - fp16 (make sure it isn't called while training)
        if not trainer.is_in_train and trainer.args.fp16_full_eval:
            model = model.half().to(trainer.args.device)

        # log prediction parameters
        logger.info(f'***** Running Prediction *****')
        logger.info(f'  Num examples = {len(test_dataset)}')
        logger.info(f'  Batch size = {dataloader.batch_size}')

        # main evaluation loop
        model.eval()
        trainer.callback_handler.eval_dataloader = dataloader
        logits_all, probs_all, labels_all = None, None, None
        for step, inputs in enumerate(dataloader):
            inputs = trainer._prepare_inputs(inputs)

            # apply inference
            logits, probs = self._generate(
                inputs['input_ids'], output_probs,
                attention_mask=inputs['attention_mask'], max_length=max_length)

            # get labels
            labels = inputs.get('labels')
            if labels is not None and labels.shape[-1] < max_length:
                labels = trainer._pad_tensors_to_max_len(labels, max_length)

            # store variables
            logits = logits.cpu()
            logits_all = logits if logits_all is None else concat(
                logits_all, logits, padding_index=self.tokenizer.pad_token_id)
            if probs is not None:
                probs = probs.cpu()
                probs_all = probs if probs_all is None else concat(
                    probs_all, probs, padding_index=self.SKIP_PROB_TOK)
            if labels is not None:
                labels = labels.cpu()
                labels_all = labels if labels_all is None else concat(
                    labels_all, labels, padding_index=self.tokenizer.pad_token_id)

            trainer.control = trainer.callback_handler.on_prediction_step(
                trainer.args, trainer.state, trainer.control)

        # convert to numpy
        logits_all = logits_all.numpy()
        if probs_all is not None:
            probs_all = probs_all.numpy()
        if labels_all is not None:
            labels_all = labels_all.numpy()

        return TranslationOutput(predictions=logits_all, label_ids=labels_all, probs=probs_all)
