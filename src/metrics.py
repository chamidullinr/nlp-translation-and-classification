import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def accuracy(pred: np.array, targ: np.array):
    if len(pred.shape) == 2:
        pred = pred.argmax(1)
    return accuracy_score(targ, pred)


def f1(pred: np.array, targ: np.array, labels=None):
    if len(pred.shape) == 2:
        if labels is None:
            labels = np.arange(pred.shape[1])
        pred = pred.argmax(1)
    return f1_score(targ, pred, labels=labels, average='macro')


def text_accuracy(a: str, b: str):
    return int(a == b)


def levenshtein_ratio(a: str, b: str):
    import stringdist
    return 1 - stringdist.levenshtein_norm(a, b)


def jaccard_index(a: str, b: str):
    label_1 = set(a.split(' ')) if not isinstance(a, list) else set(a)
    label_2 = set(b.split(' ')) if not isinstance(b, list) else set(b)
    union_len = len(label_1.union(label_2))
    intersection_len = len(label_1.intersection(label_2))
    return intersection_len / union_len


class ClassificationMetricsCallback:
    def __init__(self, metrics=[accuracy, f1]):
        self.metrics = metrics

    def __call__(self, eval_preds):
        preds, labels = eval_preds

        # compute metrics
        out = {met.__name__: met(preds, labels) for met in self.metrics}
        
        return out


class TranslationMetricsCallback:
    def __init__(self, tokenizer, metrics=[text_accuracy, levenshtein_ratio, jaccard_index]):
        self.tokenizer = tokenizer
        self.metrics = metrics

    def __call__(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # replace -100 with pad token in the labels
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        # decode preds and labels
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # post-process decoded strings
        decoded_preds = [x.strip() for x in decoded_preds]
        decoded_labels = [x.strip() for x in decoded_labels]

        # compute metrics
        out = {}
        for met in self.metrics:
            out[met.__name__] = np.mean(
                [met(a, b) for a, b in zip(decoded_preds, decoded_labels)])

        return out
