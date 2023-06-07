import re
import torch
import torchvision.transforms as T
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
from torch.nn.utils.rnn import pad_sequence
import math
from emoji import demojize
from nltk.tokenize import TweetTokenizer
import re
import logging
import urllib.request
from typing import Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoConfig,\
    AutoModelForMaskedLM
import num2words

def convert_num_to_words(utterance):
      utterance = ' '.join([num2words.num2words(i) if i.isdigit() else i for i in utterance.split()])
      return utterance

tokenizer = TweetTokenizer()

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalize(tweet):
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    return " ".join(normTweet.split())

def preprocess(sentence):
    sentence = re.sub("[hH]ttp\S*", "", sentence)    # remove url
    # sentence = re.sub("@\S*", "", sentence)           # remove @
    # sentence = sentence.replace("#", "# ")
    # sentence = re.sub("#\S*", "", sentence)             # remove #
    # sentence = re.sub(r"([A-Z][a-z]*)", r"\1", sentence)
    # sentence = re.sub("[0-9]", "", sentence)         # remove numbers
    # sentence = convert_num_to_words(sentence)
    sentence = sentence.lower()                      # convert into lowercase
    return remove_tags(sentence)

def print_metrics(y_true, y_pred):
    print(f"Confusion matrix: \n {confusion_matrix(y_true, y_pred)}")
    print(f"F1 Score (Micro): {f1_score(y_true, y_pred, average='micro')}")
    print(f"F1 Score (Macro): {f1_score(y_true, y_pred, average='macro')}")
    print(f"F1 Score (Weighted): {f1_score(y_true, y_pred, average='weighted')}")
    print(f"Accuracy): {accuracy_score(y_true, y_pred)}")

def print_metrics3(y_true, y_pred, output_path):
    labels = list(set(y_true))

    rows = []
    print(f"Confusion matrix: \n {confusion_matrix(y_true, y_pred)}")
    rows.append(accuracy_score(y_true, y_pred))

    for label in labels:
        rows.append(precision_score(y_true, y_pred, average='binary',pos_label=label))
        rows.append(recall_score(y_true, y_pred, average='binary',pos_label=label))
        rows.append(f1_score(y_true, y_pred, average='binary',pos_label=label))
    with open(output_path, 'a', encoding="utf-8") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(rows)


def print_metrics2(y_true, y_pred):


    labels = list(set(y_true))

    print(f"Confusion matrix: \n {confusion_matrix(y_true, y_pred)}")
    print(f"Accuracy): {accuracy_score(y_true, y_pred)}")

    for label in labels:
        print(f"Precision ({label}): {precision_score(y_true, y_pred, average='binary',pos_label=label)}")
        print(f"Recall ({label}): {recall_score(y_true, y_pred, average='binary',pos_label=label)}")
        print(f"F1 Score ({label}): {f1_score(y_true, y_pred, average='binary',pos_label=label)}")

def labels_to_weights2(labels, labels2):
    num = max(labels) + 1
    counts = [labels.count(i)+1 for i in range(0, num)]
    total = sum(counts)
    counts2 = [labels2.count(i)+1 for i in range(0, num)]
    total2 = sum(counts2)
    counts = [counts2[idx]/counts[idx] for idx , count in enumerate(counts)]
    return torch.tensor(counts, dtype=torch.float)

def labels_to_weights(labels):
    num = max(labels) + 1
    counts = [labels.count(i)+1 for i in range(0, num)]
    counts = [1/(count) for count in counts]    
    total = sum(counts)
    counts = [3 * count/total for count in counts]    

    return torch.tensor(counts, dtype=torch.float)


def image_transforms():
    transforms = []
    transforms.append(T.Resize(256))
    transforms.append(T.CenterCrop(224))
    return T.Compose(transforms)

def label_to_value(label):
    VALUES = [0.0,-1.0, 1.0]
    value = VALUES[int(label)]
    value = np.asarray([value])
    value = torch.tensor(value, dtype=torch.float)
    return value
    
def cosine_distance(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def length_to_mask(length, max_len=None, dtype=None):
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    mask = ~mask
    return mask

def pad_tensor(tensor, batch_index):
    num_seq = torch.unique(batch_index)
    tensors = [tensor[batch_index == seq_id] for seq_id in num_seq]
    lengths = [len(tensor) for tensor in tensors]
    lengths = torch.tensor(lengths).to(num_seq.device)
    masks = length_to_mask(lengths)
    return pad_sequence(tensors, batch_first=True), masks.bool()