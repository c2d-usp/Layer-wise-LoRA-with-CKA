import os
import argparse
from typing import List, Optional, Tuple, Sequence

import torch
from datasets import load_dataset, Image as HFImage, DatasetDict
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
)
from math import floor
import numpy as np

LETTER_MAP = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

PRE_PROMPT = ""
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."


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

def extract_representations_from_multimodal_model(
        model,
        tokenizer,
        train_dataset,
        device,
        batch_size = 1,
        layers_to_extract = [-1],
        feature_selection = "last_token") -> Tuple[torch.Tensor,...]:
    
    all_features = {layer: [] for layer in layers_to_extract}

    for i in range(0,len(train_dataset), batch_size):
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in train_dataset.select(range(i,i + batch_size))]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in train_dataset.select(range(i,i + batch_size))]
        max_len = max(x.size(0) for x in input_ids)

        def pad(seq_list, pad_value):
            out = torch.full((len(seq_list), max_len), pad_value, dtype=seq_list[0].dtype)
            for i, s in enumerate(seq_list):
                out[i, : s.size(0)] = s
            return out

        batch_input_ids = pad(input_ids, tokenizer.pad_token_id)
        batch_labels = pad(labels, -100)
        batch_attn = (batch_input_ids != tokenizer.pad_token_id).long()

        pixel_values = torch.stack([torch.tensor(f["pixel_values"]) for f in train_dataset.select(range(i,i + batch_size))], dim=0)

        with torch.no_grad():
            model_outputs = model(
                input_ids=batch_input_ids.to(device),
                pixel_values=pixel_values.to(device),
                labels=batch_labels.to(device),
                attention_mask=batch_attn.to(device),
                output_hidden_states=True
            )
        


        for layer in layers_to_extract:
                if feature_selection == "last_token":
                    hidden_states_layer = model_outputs.hidden_states[layer] 
                    
                    sequence_lengths = batch_attn.sum(dim=1)
                    
                    last_token_indices = sequence_lengths - 1
                    
                    last_token_activations = []
                    for i in range(hidden_states_layer.size(0)): 
                        last_idx = last_token_indices[i]
                        last_token_activations.append(hidden_states_layer[i, last_idx, :])
                    
                    activations = torch.stack(last_token_activations, dim=0)
                all_features[layer].append(activations.cpu())
        
        del input_ids, batch_attn, batch_input_ids, pixel_values, model_outputs
        torch.cuda.empty_cache()  
        
    for layer in layers_to_extract:
        all_features[layer] = torch.cat(all_features[layer], dim=0)

    return all_features

def one_hot_encoded_to_layer_index_list(one_hot_encoded_list):
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

def build_mc_prompt(
    question: str,
    choices: List[str],
    hint: Optional[str],) -> str:
    """
    Build a multiple-choice prompt for LLaVA.
    """
    choices_str = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))

    ctx = f"Context: {hint}\n" if hint and str(hint).strip() else ""
    return f"{PRE_PROMPT}{ctx}{question}\n{choices_str}{POST_PROMPT}"


def cast_image_column(ds):
    """
    Cast the image column to HFImage type.
    """
    if "image" in ds.column_names and not isinstance(ds.features["image"], HFImage):
        ds = ds.cast_column("image", HFImage())
    return ds

def has_image(example):
    """
    Returns True if the example has an image.
    """
    return example.get("image") is not None


def preprocess_example(example, processor):
    """
    Preprocess a single example for LLaVA fine-tuning. Returns input_ids, labels, and pixel_values.
    """
    tokenizer = processor.tokenizer

    question = example["question"]
    choices = example["choices"]
    hint = example.get("hint", None)

    ans_idx = int(example["answer"]) if not isinstance(example["answer"], list) else int(example["answer"][0])
    gold_letter = LETTER_MAP.get(ans_idx, "A")

    image = example.get("image", None)
    mc_text = build_mc_prompt(question, choices, hint)

    conversation = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": mc_text}],
    }]

    text_prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

    enc = processor(text=[text_prompt], images=[image], return_tensors="pt")
    input_ids = enc["input_ids"][0]
    pixel_values = enc["pixel_values"][0]

    answer_ids = tokenizer(gold_letter, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    answer_ids = torch.tensor(answer_ids, dtype=torch.long)

    input_ids_full = torch.cat([input_ids, answer_ids], dim=0)

    labels = torch.full_like(input_ids_full, fill_value=-100)
    labels[-len(answer_ids):] = answer_ids

    out = {
        "input_ids": input_ids_full.tolist(),
        "labels": labels.tolist(),
        "pixel_values": pixel_values.numpy(),
    }
    return out

def sort_layers_by_cka(cka_layer_list: Sequence[float]) -> Sequence[int]:
    """
    Sort layers by their CKA. Receive a list of CKA values between consecutive layers and return the layer indices sorted by CKA.
    """

    max_dif_layers = sorted(range(len(cka_layer_list)), key=lambda i: cka_layer_list[i])
    layers_to_finetune_list = max_dif_layers
    
    return layers_to_finetune_list

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--dataset_name", type=str, default="derek-thomas/ScienceQA")
    parser.add_argument("--number_of_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument('--ratio_of_layers',type=float, default=0.5)

    args = parser.parse_args()

    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device
    )

    base_processor_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(base_processor_id)
    tokenizer = processor.tokenizer
    if getattr(processor, "chat_template", None) is None and getattr(tokenizer, "chat_template", None) is not None:
        processor.chat_template = tokenizer.chat_template
    if getattr(processor, "chat_template", None) is None:
        raise ValueError(
            f"No chat_template found on processor/tokenizer loaded from {base_processor_id}. "
            "Update transformers to >=4.44 and ensure the base model repo contains a chat_template."
        )
    tokenizer.pad_token = tokenizer.eos_token

    
    raw = load_dataset(args.dataset_name)
    if isinstance(raw, DatasetDict) and "train" in raw:
        ds_train = raw["train"]
    else:
        ds_train = raw["train"]

    ds_train = cast_image_column(ds_train).filter(has_image)

    def _map_fn(ex): return preprocess_example(ex, processor)
    keep_cols = ("image", "question", "choices", "answer", "hint")
    ds_train_pp = ds_train.map(_map_fn, remove_columns=[c for c in ds_train.column_names if c in keep_cols])

    ds_train_pp = ds_train_pp.select(range(args.number_of_samples))

    reps = extract_representations_from_multimodal_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_train_pp,
        device=args.device,
        batch_size=args.batch_size,
        layers_to_extract=list(range(len(model.language_model.layers) + 1)),
        feature_selection="last_token"
    )

    base_out = os.path.join(args.output_dir,"layer_selection")
    os.makedirs(base_out, exist_ok=True)

    number_of_layers = len(model.language_model.layers)

    cka_layer_list = []
    for i in range(0,number_of_layers):
        cka_layer_list.append((linear_cka(reps[i],reps[i + 1])))

    with open(f"{base_out}/cka_layer_list_{str(args.model_name_or_path).split('/')[-1]}_{args.number_of_samples}.txt", "w") as file:
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

    with open(f"{base_out}/layers_to_finetune_ohe_{str(args.model_name_or_path).split('/')[-1]}_{args.number_of_samples}.txt", "w") as file:
        file.write(one_hot_string)

    


if __name__ == "__main__":
    main()