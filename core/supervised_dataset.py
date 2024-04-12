from dataclasses import dataclass
import json
import os
from typing import Dict, Sequence
from torch.utils.data import Dataset
import datasets
import logging
import torch.distributed as dist
import torch
import transformers
import copy
import math


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<|pad|>"
DEFAULT_EOS_TOKEN = "<|endoftext|>"
DEFAULT_UNK_TOKEN = "<|unk|>"


def fmt_prompt(prompt):
    return f"{prompt}"


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _filter_tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    samples = []
    for text in strings:
        tokens = tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        if tokens.input_ids.squeeze().numel() < tokenizer.model_max_length:
            samples.append(True)
        else:
            samples.append(False)

    return samples


def filter_long_samples(
    samples: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    sources = [f"{fmt_prompt(question)}" for question in samples["input"]]
    targets = [f"{answer}{tokenizer.eos_token}" for answer in samples["label"]]
    examples = [s + t for s, t in zip(sources, targets)]

    return _filter_tokenize_fn(examples, tokenizer)


# class SupervisedDataset(Dataset):
#     """Dataset for supervised fine-tuning."""
#     def __init__(
#         self,
#         tokenizer: transformers.PreTrainedTokenizer,
#         data_paths,
#         limit=None,
#     ):
#         super(SupervisedDataset, self).__init__()
#         workers = math.ceil(os.cpu_count() / dist.get_world_size())
#         logging.warning(f"TOKENIZING WITH NUM_WORKERS: {workers}")
#         dataset = (
#             datasets.load_dataset(
#                 "json",
#                 data_files=data_paths,
#                 split=f"train[0:{limit}]" if limit else "train",
#             )
#             .filter(
#                 lambda samples: filter_long_samples(samples, tokenizer),
#                 batched=True,
#                 batch_size=3000,
#                 num_proc=workers,
#             )
#             .map(
#                 lambda samples: preprocess(samples, tokenizer),
#                 batched=True,
#                 batch_size=3000,
#                 num_proc=workers,
#             )
#         )
#         self.input_ids = dataset["input_ids"]
#         self.labels = dataset["labels"]

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         return dict(
#             input_ids=torch.tensor(self.input_ids[i]),
#             labels=torch.tensor(self.labels[i]),
#         )

import torch.nn.functional as F


# def preprocess(
#     samples: Sequence[str],
#     tokenizer: transformers.PreTrainedTokenizer,
# ) -> Dict:
#     """Preprocess the data by tokenizing."""
#     input_ids = []
#     labels = []
#     for es_input, es_label in zip(samples["input"], samples["label"]):
#         tokens = tokenizer(
#             es_input,
#             return_tensors="pt",
#             padding=False,
#             max_length=tokenizer.model_max_length,
#             truncation=True,
#         )
#         shape_input_ids = tokens.input_ids.size()
#         if es_label == 1:
#             temp_label = torch.ones(shape_input_ids).long()
#         elif es_label == 0:
#             temp_label = torch.zeros(shape_input_ids).long()
#         else:
#             raise ValueError(f"Invalid label value: {es_label}")
#         input_ids.append(tokens.input_ids[0])
#         labels.append(temp_label)
#     return dict(input_ids=input_ids, labels=labels)


# def preprocess(
#     samples: Sequence[str],
#     tokenizer: transformers.PreTrainedTokenizer,
# ) -> Dict:
#     """Preprocess the data by tokenizing."""
#     input_ids = []
#     labels = []
#     for es_input, es_label in zip(samples["input"], samples["label"]):
#         tokens = tokenizer(
#             es_input,
#             return_tensors="pt",
#             padding=False,
#             max_length=tokenizer.model_max_length,
#             truncation=True,
#         )
#         if es_label == 1:
#             label = torch.tensor(1.0)  # Convert to float tensor
#         elif es_label == 0:
#             label = torch.tensor(0.0)  # Convert to float tensor
#         else:
#             # Skip samples with invalid labels or assign a default value
#             continue
#         input_ids.append(tokens.input_ids[0])
#         labels.append(label)
#     return dict(input_ids=input_ids, labels=labels)

# # @dataclass
# # class DataCollatorForSupervisedDataset(object):
#     """Collate examples for supervised fine-tuning."""
#     tokenizer: transformers.PreTrainedTokenizer

#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         input_ids, labels = tuple(
#             [instance[key] for instance in instances] for key in ("input_ids", "labels")
#         )

#         # Check if input_ids are single-dimensional tensors
#         if input_ids[0].dim() == 1:
#             input_ids = [ids.unsqueeze(0) for ids in input_ids]

#         # Get the maximum sequence length in the batch
#         max_length = max(ids.size(1) for ids in input_ids)
#         print(max_length)

#         # Pad the input_ids to the maximum sequence length
#         input_ids = [F.pad(ids, (0, max_length - ids.size(1)), value=self.tokenizer.pad_token_id) for ids in input_ids]

#         # Pad the labels to the maximum sequence length
#         labels = [F.pad(label, (0, max_length - label.size(1)), value=IGNORE_INDEX) for label in labels]

#         input_ids = torch.cat(input_ids, dim=0)
#         labels = torch.cat(labels, dim=0)

#         return dict(
#             input_ids=input_ids,
#             labels=labels,
#             attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
#         )

# @dataclass
# class DataCollatorForSupervisedDataset(object):
#     """Collate examples for supervised fine-tuning."""
#     tokenizer: transformers.PreTrainedTokenizer

#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         input_ids, labels = tuple(
#             [instance[key] for instance in instances] for key in ("input_ids", "labels")
#         )
#         # Check if input_ids are single-dimensional tensors
#         if input_ids[0].dim() == 1:
#             input_ids = [ids.unsqueeze(0) for ids in input_ids]
#         # Get the maximum sequence length in the batch
#         max_length = max(ids.size(1) for ids in input_ids)
#         # Pad the input_ids to the maximum sequence length
#         input_ids = [F.pad(ids, (0, max_length - ids.size(1)), value=self.tokenizer.pad_token_id) for ids in input_ids]
#         input_ids = torch.cat(input_ids, dim=0)
#         labels = torch.tensor(labels)  # Convert labels to a tensor
#         return dict(
#             input_ids=input_ids,
#             labels=labels,
#             attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
#         )

def filter_long_samples(
    samples: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    inputs = [f"{input_text}{tokenizer.eos_token}" for input_text in samples["input"]]
    labels = [f"{label_text}{tokenizer.eos_token}" for label_text in samples["label"]]
    
    input_lengths = _filter_tokenize_fn(inputs, tokenizer)
    label_lengths = _filter_tokenize_fn(labels, tokenizer)
    
    return [input_len and label_len for input_len, label_len in zip(input_lengths, label_lengths)]


def preprocess(
    samples: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    inputs = [f"{input_text}{tokenizer.eos_token}" for input_text in samples["input"]]
    labels = [f"{label_text}{tokenizer.eos_token}" for label_text in samples["label"]]
    
    inputs_tokenized = _tokenize_fn(inputs, tokenizer)
    labels_tokenized = _tokenize_fn(labels, tokenizer)
    
    return dict(input_ids=inputs_tokenized["input_ids"], labels=labels_tokenized["input_ids"])

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_paths,
        limit=None,
    ):
        super(SupervisedDataset, self).__init__()
        self.input_ids = []
        self.labels = []

        for data_path in data_paths:
            with open(data_path, 'r') as file:
                for line in file:
                    try:
                        example = json.loads(line.strip())
                        input_text = example['input']
                        label_text = example['label']

                        input_ids = tokenizer.encode(input_text, add_special_tokens=False)
                        label_ids = tokenizer.encode(label_text, add_special_tokens=False)

                        self.input_ids.append(input_ids)
                        self.labels.append(label_ids)

                        if limit is not None and len(self.input_ids) >= limit:
                            break
                    except json.decoder.JSONDecodeError:
                        print(f"Skipping invalid JSON line: {line}")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=torch.tensor(self.input_ids[i]),
            labels=torch.tensor(self.labels[i]),
        )



@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )