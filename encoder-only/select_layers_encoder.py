 # This file contains a function adapted from:
 # "call_sequence_classification_model" by Max Klabunde, licensed under CC-BY 4.0.
 # Original source: https://github.com/mklabunde/resi
 # Modified by Keith Ando (2025).

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, HfArgumentParser
from datasets import load_dataset
from math import floor
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, Sequence

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

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

def sort_layers_by_cka(cka_layer_list: Sequence[float]) -> Sequence[int]:
    """
    Sort layers by their CKA. Receive a list of CKA values between consecutive layers and return the layer indices sorted by CKA.
    """

    max_dif_layers = sorted(range(len(cka_layer_list)), key=lambda i: cka_layer_list[i])
    layers_to_finetune_list = max_dif_layers
    
    return layers_to_finetune_list

@dataclass
class Arguments:
    model_name_or_path: str = field(default= "roberta-base")
    glue_task: str = field(default = "rte")
    number_of_samples: Optional[int] = field(default=None)
    device: str = field(default="cuda")
    output_dir: str = field(default=".")
    batch_size: int = field(default=256)
    ratio_of_layers : float = field(default=0.5)

def extract_from_sequence_classification_model(
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        prompt: Sequence[str],
        device: torch.device,
        max_length: int,
        with_text_pair = False,
        padding_type = "max_length",
        batch_size = 1,
        layers_to_extract = [-1],
        feature_selection = "cls_token") -> Tuple[torch.Tensor,...]:
    """
    Given a dataset, tokenizer, and model, extract hidden representations from specified layers.
    The output is a dictionary mapping layer indices to their corresponding representations.
    """

    all_features = {layer: [] for layer in layers_to_extract}
 
    for i in range(0, len(prompt),batch_size):
            batch_prompts = prompt[i:i+batch_size]

            if with_text_pair == True:
                model_inputs = tokenizer(text = [text[0] for text in batch_prompts],
                                        text_pair = [text[1] for text in batch_prompts], 
                                        return_tensors="pt",
                                        max_length = max_length,
                                        padding = padding_type,
                                        truncation = True).to(device)
            else:
                model_inputs = tokenizer(batch_prompts, 
                                    return_tensors="pt",
                                    max_length = max_length,
                                    padding = padding_type,
                                    truncation = True).to(device)
                
            input_ids = model_inputs["input_ids"].to(device)
            attention_mask = model_inputs["attention_mask"].to(device)  
            with torch.no_grad():
                model_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            for layer in layers_to_extract:
                if feature_selection == "cls_token":
                    activations = model_outputs.hidden_states[layer][:, 0, :]
                else:
                    raise ValueError("Undefined Token Extraction Method")
                all_features[layer].append(activations.cpu()) 

    for layer in layers_to_extract:
        all_features[layer] = torch.cat(all_features[layer], dim=0)

    return all_features

def main():
    parser = HfArgumentParser(Arguments)

    args = parser.parse_args_into_dataclasses()[0]
    dataset = load_dataset("nyu-mll/glue", args.glue_task)

    base_out = os.path.join(args.output_dir,"layer_selection")
    os.makedirs(base_out, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
         
    if args.number_of_samples is not None:
        dataset["train"] = dataset["train"].select(range(args.number_of_samples))
    

    sentence1_key, sentence2_key = task_to_keys[args.glue_task]


    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is None:

        base = getattr(model, getattr(model, "base_model_prefix", ""), model)
        if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
            num_layers = len(base.encoder.layer)
        elif hasattr(base, "encoder") and hasattr(base.encoder, "layers"):
            num_layers = len(base.encoder.layers)
        elif hasattr(base, "transformer") and hasattr(base.transformer, "h"):
            num_layers = len(base.transformer.h)
        else:
            raise RuntimeError("Could not determine number of transformer layers from the model.")

    layers_to_extract = list(range(num_layers + 1))
    print(f"Detected num_hidden_layers={num_layers}, layers_to_extract={layers_to_extract}")


    if sentence2_key == None:
        features_map = extract_from_sequence_classification_model(model,tokenizer,[element for element in dataset["train"][sentence1_key]],device = args.device, max_length = 512,batch_size=args.batch_size,layers_to_extract=layers_to_extract)
    else:
        features_map = extract_from_sequence_classification_model(model,tokenizer,[[element[sentence1_key],element[sentence2_key]] for element in dataset["train"]],device = args.device, max_length = 512,batch_size=args.batch_size,layers_to_extract=layers_to_extract,with_text_pair = True)
    
    cka_layer_list = []
    for i in range(0,num_layers):
        cka_layer_list.append((linear_cka(features_map[i],features_map[i + 1])))

    with open(f"{base_out}/cka_layer_list_{str(args.model_name_or_path).split('/')[-1]}_{args.glue_task}.txt", "w") as file:
        for i in range(0,num_layers):
            file.write(f"{cka_layer_list[i]}" + "|" + f"{i},{i + 1}\n")

    cka_layer_idx_to_finetune_sorted = sort_layers_by_cka(cka_layer_list)
    
    print(f"All {num_layers} layers, sorted by CKA: {cka_layer_idx_to_finetune_sorted}")

    number_of_layers_to_select = int(floor(args.ratio_of_layers*num_layers))
    selected_layers_to_finetune = cka_layer_idx_to_finetune_sorted[1:number_of_layers_to_select + 1]

    print(f"Top {number_of_layers_to_select} selected layers: {selected_layers_to_finetune}")

    one_hot = [
    1 if i in selected_layers_to_finetune else 0
    for i in range(num_layers)
    ]
    one_hot_string = "".join(map(str, one_hot))

    print(f"One-hot encoded list: {one_hot_string}")

    with open(f"{base_out}/layers_to_finetune_ohe_{str(args.model_name_or_path).split('/')[-1]}_{args.glue_task}.txt", "w") as file:
        file.write(one_hot_string)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()