 # This file contains a function adapted from:
 # "call_causal_lm_model" by Max Klabunde, licensed under CC-BY 4.0.
 # Original source: https://github.com/mklabunde/resi
 # Modified by Keith Ando (2025).


import os
import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer
)
from datasets import (
     concatenate_datasets,
     load_dataset
)
from typing import (
     Dict, 
     Sequence,
     Tuple
)
from copy import (
    deepcopy
)
from math import (
    floor
)

PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )

IGNORE_INDEX = -100

def linear_cka(
    features_x: torch.Tensor,
    features_y: torch.Tensor
) -> float:
    """
    Compute linear CKA (centered kernel alignment) between two feature matrices (n x d_x) and (n x d_y).

    """

    if features_x.ndim != 2 or features_y.ndim != 2:
        raise ValueError("features must be 2D (n_examples x dim)")
    features_x = features_x.float()
    features_y = features_y.float()

    x = features_x - features_x.mean(dim=0, keepdim=True)
    y = features_y - features_y.mean(dim=0, keepdim=True)

    k_xy = x.t().mm(y)   
    k_xx = x.t().mm(x)   
    k_yy = y.t().mm(y)   

    dot_xy = torch.norm(k_xy).pow(2) 
    norm_x = torch.norm(k_xx)
    norm_y = torch.norm(k_yy)

    denom = (norm_x * norm_y).clamp(1e-12)
    cka_val = (dot_xy / denom).item()
    return float(cka_val)

def _tokenize_fn(strings: Sequence[str], tokenizer: PreTrainedTokenizer) -> Dict:
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
    tokenizer: PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    targets = [f"{output}\n{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def extract_representations_from_autoregressive_model(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        train_dataset: Dict,
        device: torch.device,
        batch_size = 1,
        layers_to_extract = [-1],
        feature_selection = "last_token") -> Tuple[torch.Tensor,...]:
    """
    Given a dataset, tokenizer, and model, extract hidden representations from specified layers.
    The output is a dictionary mapping layer indices to their corresponding representations.
    """

    all_features = {layer: [] for layer in layers_to_extract}

    for i in range(0,len(train_dataset), batch_size):
        input_ids = train_dataset[i:i + batch_size]["input_ids"]
        labels = train_dataset[i:i + batch_size]["labels"]
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )

        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            model_outputs = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
    
        for layer in layers_to_extract:
                if feature_selection == "last_token":
                    hidden_states_layer = model_outputs.hidden_states[layer]
                    sequence_lengths = attention_mask.sum(dim=1) 
                    last_token_indices = sequence_lengths - 1
                    
                    last_token_activations = []

                    for i in range(hidden_states_layer.size(0)): 
                        last_idx = last_token_indices[i]
                        last_token_activations.append(hidden_states_layer[i, last_idx, :])
                    
                    activations = torch.stack(last_token_activations, dim=0)

                all_features[layer].append(activations.cpu())
        
        del input_ids, attention_mask, model_outputs
        torch.cuda.empty_cache()  

    for layer in layers_to_extract:
        all_features[layer] = torch.cat(all_features[layer], dim=0)

    return all_features

def sort_layers_by_cka(cka_layer_list: Sequence[float]) -> Sequence[int]:
    """
    Sort layers by their CKA. Receive a list of CKA values between consecutive layers and return the layer indices sorted by CKA.
    """

    max_dif_layers = sorted(range(len(cka_layer_list)), key=lambda i: cka_layer_list[i])
    layers_to_finetune_list = max_dif_layers
    
    return layers_to_finetune_list

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_path', type=str, default = 'meta-llama/Llama-2-7b-hf')
    parser.add_argument('--task', type=str, default = 'metamath')
    parser.add_argument('--number_of_samples', type=int, default= 128)
    parser.add_argument('--device', type=str,default="auto")
    parser.add_argument('--output_dir', type=str, default=".")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--ratio_of_layers',type=float, default=0.5)

    args = parser.parse_args()

    base_out = os.path.join(args.output_dir,"layer_selection")
    os.makedirs(base_out, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map = args.device,
        torch_dtype = torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            model_max_length=512,
            padding_side="right",
            use_fast=True,
    )
    
    tokenizer.pad_token = tokenizer.eos_token

    all_training_dataset = []
    sub_task = [f"{args.task}:{args.number_of_samples}"]

    for task in sub_task:
            if ":" in task: 
                cur_task, num_split = task.split(":")
                cur_split = f"train[:{num_split}]"
            else:
                cur_task, cur_split = task, "train"

            ds = load_dataset("fxmeng/pissa-dataset", data_dir=cur_task, split=cur_split)
            all_training_dataset.append(ds)
            
    raw_train_datasets = concatenate_datasets(all_training_dataset)
    train_dataset = raw_train_datasets.map(
            train_tokenize_function,
            batched=True,
            batch_size=3000,
            num_proc=32,
            remove_columns=raw_train_datasets.column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on train dataset",
            fn_kwargs={"tokenizer": tokenizer, "query": "instruction", "response": "output"}
        )
    
    number_of_layers = len(model.model.layers)
    layers_to_extract = list(range(number_of_layers + 1))

    reps = extract_representations_from_autoregressive_model(model = model, 
                                                             tokenizer = tokenizer, 
                                                             train_dataset = train_dataset, 
                                                             device = model.device,
                                                             batch_size = args.batch_size,
                                                             layers_to_extract = layers_to_extract)

    cka_layer_list = []
    for i in range(0,number_of_layers):
        cka_layer_list.append((linear_cka(reps[i],reps[i + 1])))

    with open(f"{base_out}/cka_layer_list_{str(args.model_name_or_path).split('/')[-1]}_{args.task}_{args.number_of_samples}.txt", "w") as file:
        for i in range(0,number_of_layers):
            file.write(f"{cka_layer_list[i]}" + "|" + f"{i},{i + 1}\n")

    cka_layer_idx_to_finetune_sorted = sort_layers_by_cka(cka_layer_list)
    
    print(f"All {number_of_layers} layers, sorted by CKA: {cka_layer_idx_to_finetune_sorted}")

    number_of_layers_to_select = int(floor(args.ratio_of_layers*number_of_layers))
    selected_layers_to_finetune = cka_layer_idx_to_finetune_sorted[:number_of_layers_to_select]

    print(f"Top {number_of_layers_to_select} selected layers: {selected_layers_to_finetune}")

    one_hot = [
    1 if i in selected_layers_to_finetune else 0
    for i in range(number_of_layers)
    ]
    one_hot_string = "".join(map(str, one_hot))

    print(f"One-hot encoded list: {one_hot_string}")

    with open(f"{base_out}/layers_to_finetune_ohe_{str(args.model_name_or_path).split('/')[-1]}_{args.task}_{args.number_of_samples}.txt", "w") as file:
        file.write(one_hot_string)

if __name__ == "__main__":
    main()