# This file is adapted from the PiSSA repository:
# https://github.com/GraphPKU/PiSSA
#
# Modifications:
# - Adapted by Keith Ando (2025).
# - Extended the original implementation with Layer-wise LoRA fine-tuning,
#   enabling selective fine-tuning of a subset of transformer layers.

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Union

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from transformers import set_seed
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, concatenate_datasets

IGNORE_INDEX = -100
logger = logging.getLogger(__name__)

PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    seed: int = field(default=0, metadata={"help": "Random seed for initialization."})
    # Model Arguments
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-2-7b-hf")
    attn_implementation : Optional[str] = field(default="flash_attention_2")

    # LoRA Settings
    lora_rank: Optional[int] = field(default=8)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.0)
    init_weights: Union[str] = field(default="standard",metadata={"help": ("standard -> LoRA; `pissa` -> PiSSA; `pissa_niter_16` -> Fast SVD PiSSA"),},)
    target_modules : Optional[str] = field(default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj")
    lora_layers: str = field(
            default=None,
            metadata={"help":"Layers to Apply LoRA modules, use this parameter as a one hot encoded list. If there is five layers, and you want to apply LoRA to layer 5 and 1, the argument would be 10001"}
        )
    # Data Arguments
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    sub_task: List[str] = field(default=None)
    dataset_split: str = field(default="train", metadata={"help": "(`['train', 'test', 'eval']`):"})
    dataset_field: List[str] = field(default=None, metadata={"help": "Fields of dataset input and output."})
    shuffle_dataset : Optional[bool] = field(default=False)
    # Training Arguments
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

def one_hot_str_to_layer_indices(one_hot_encoded_list):
    """
    Transform one-hot-encoded string to list of layers indices.
    """
    layer_index_list = []
    index = 0
    while index < len(one_hot_encoded_list):
        if one_hot_encoded_list[index] == '1':
            layer_index_list.append(index)
        index += 1
    return layer_index_list

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    
def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    targets = [f"{output}\n{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_path,args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    all_training_dataset = []
    for task in args.sub_task:
        if ":" in task: # e.g. math:500, gsm8k:100
            cur_task, num_split = task.split(":")
            cur_split = f"{args.dataset_split}[:{num_split}]"
        else:
            cur_task, cur_split = task, args.dataset_split

        ds = load_dataset(args.data_path, data_dir=cur_task, split=cur_split)
        all_training_dataset.append(ds)
        
    raw_train_datasets = concatenate_datasets(all_training_dataset)
    if args.shuffle_dataset:
        
        raw_train_datasets = raw_train_datasets.shuffle(seed=args.seed)
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={"tokenizer": tokenizer, "query": args.dataset_field[0], "response": args.dataset_field[1]}
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def train(args):
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        device_map = "auto",
        torch_dtype=torch.bfloat16,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for param in model.parameters():
        param.requires_grad = False

    if args.init_weights == "pissa":
        init_weights_config = "pissa"
    elif args.init_weights == "standard":
        init_weights_config = True

    if args.lora_layers != None: 
        lora_config = LoraConfig(r = args.lora_rank,
                                    lora_alpha = args.lora_alpha,
                                    lora_dropout = args.lora_dropout,
                                    task_type=TaskType.CAUSAL_LM,
                                    target_modules=args.target_modules.split(','),
                                    init_lora_weights=init_weights_config,
                                    layers_to_transform=one_hot_str_to_layer_indices(args.lora_layers)  
                                    )
    else:
        lora_config = LoraConfig(r = args.lora_rank,
                                    lora_alpha = args.lora_alpha,
                                    lora_dropout = args.lora_dropout,
                                    target_modules=args.target_modules.split(','),
                                    init_lora_weights=init_weights_config,
                                    task_type=TaskType.CAUSAL_LM,
                                    )

    logger.info(f"Using LoRA config: {lora_config}")

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


    data_module = make_supervised_data_module(tokenizer=tokenizer, data_path=args.data_path, args = args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=args, **data_module)
    trainer.train()

    model = model.merge_and_unload()
    model.save_pretrained(args.output_dir + "/model")
    tokenizer.save_pretrained(args.output_dir + "/tokenizer")
    logging.warning("Training finished, model and tokenizer saved to output directory.")

if __name__ == "__main__":
    parser = transformers.HfArgumentParser((TrainingArguments))
    args = parser.parse_args_into_dataclasses()[0]
    set_seed(args.seed)
    train(args)
