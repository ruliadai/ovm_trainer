from dataclasses import dataclass
import os
from typing import Counter, Dict, Sequence
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
    return f"### Instructions:\n{prompt}\n\n### Response:"


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


def preprocess(
    train_on_inputs: bool,
    samples: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    sources = [f"{question}" for question in samples["input"]]
    examples = [s for s in sources]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = [float(torch.tensor(label)) for label in samples["label"]]

    # labels = copy.deepcopy(input_ids)
    # for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
    #     if not train_on_inputs:
    #         label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


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
    sources = [f"{question}" for question in samples["input"]]
    examples = [s for s in sources]

    return _filter_tokenize_fn(examples, tokenizer)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        train_on_inputs: bool,
        tokenizer: transformers.PreTrainedTokenizer,
        data_paths,
        limit=None,
    ):
            super(SupervisedDataset, self).__init__()
        # try:
            workers = math.ceil(os.cpu_count() / dist.get_world_size())
            logging.warning(f"TOKENIZING WITH NUM_WORKERS: {workers}")
            dataset = (
                datasets.load_dataset(
                    "json",
                    data_files=data_paths,
                    split=f"train[0:{limit}]" if limit else "train",
                )
                .filter(
                    lambda samples: filter_long_samples(samples, tokenizer),
                    batched=True,
                    batch_size=3000,
                    num_proc=workers,
                )
                .map(
                    lambda samples: preprocess(train_on_inputs, samples, tokenizer),
                    batched=True,
                    batch_size=3000,
                    num_proc=workers,
                )
            )

            self.input_ids = dataset["input_ids"]
            self.labels = dataset["labels"]
            # Calculate pos_weights and neg_weights
            label_counts = Counter(self.labels)

            total_samples = len(self.labels)
            self.pos_weights =   label_counts[1]/total_samples if label_counts[1] > 0 else 0.0
            self.neg_weights =   label_counts[0]/total_samples if label_counts[0] > 0 else 0.0


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
        
        labels = torch.tensor(labels)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


# from dataclasses import dataclass
# import os
# from typing import Dict, Sequence
# from torch.utils.data import Dataset
# import datasets
# import logging
# import torch.distributed as dist
# import torch
# import transformers
# import copy
# import math

# IGNORE_INDEX = -100
# DEFAULT_PAD_TOKEN = "<|pad|>"
# DEFAULT_EOS_TOKEN = "<|endoftext|>"
# DEFAULT_UNK_TOKEN = "<|unk|>"

# def fmt_prompt(prompt):
#     return f"### Instructions:\n{prompt}\n\n### Response:"

# def _tokenize_fn(
#     strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
# ) -> Dict:
#     """Tokenize a list of strings."""
#     tokenized_list = [
#         tokenizer(
#             text,
#             return_tensors="pt",
#             padding=False,
#             max_length=tokenizer.model_max_length,
#             truncation=True,
#         )
#         for text in strings
#     ]
#     input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
#     attention_mask = [tokenized.attention_mask[0] for tokenized in tokenized_list]
#     return dict(input_ids=input_ids, attention_mask=attention_mask)

# def preprocess(
#     samples: Sequence[str],
#     tokenizer: transformers.PreTrainedTokenizer,
# ) -> Dict:
#     """Preprocess the data by tokenizing."""
#     sources = [f"{question}" for question in samples["input"]]
#     print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
#     tokenized_inputs = _tokenize_fn(sources, tokenizer)
#     labels = [torch.tensor(label) for label in samples["label"]]
#     return dict(input_ids=tokenized_inputs["input_ids"], attention_mask=tokenized_inputs["attention_mask"], labels=labels)



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
#             .map(
#                 lambda samples: preprocess(samples, tokenizer),
#                 batched=True,
#                 batch_size=3000,
#                 num_proc=workers,
#             )
#         )
#         self.input_ids = dataset["input_ids"]
#         self.attention_mask = dataset["attention_mask"]
#         self.labels = dataset["labels"]

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         return dict(
#             input_ids=torch.tensor(self.input_ids[i]),
#             attention_mask=torch.tensor(self.attention_mask[i]),
#             label=self.labels[i],
#         )
# @dataclass
# class DataCollatorForSupervisedDataset(object):
#     """Collate examples for supervised fine-tuning."""
#     tokenizer: transformers.PreTrainedTokenizer

#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         input_ids, attention_mask, labels = tuple(
#             [instance[key] for instance in instances] for key in ("input_ids", "attention_mask", "label")
#         )
#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
#         )
#         attention_mask = torch.nn.utils.rnn.pad_sequence(
#             attention_mask, batch_first=True, padding_value=0
#         )
#         labels = torch.stack(labels)
#         return dict(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=labels,
#         )