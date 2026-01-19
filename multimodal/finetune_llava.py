import os
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from datasets import load_dataset, Image as HFImage, DatasetDict
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

LETTER_MAP = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

PRE_PROMPT = ""
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."

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


@dataclass
class LLaVADataCollator:
    pad_token_id: int
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        max_len = max(x.size(0) for x in input_ids)

        def pad(seq_list, pad_value):
            out = torch.full((len(seq_list), max_len), pad_value, dtype=seq_list[0].dtype)
            for i, s in enumerate(seq_list):
                out[i, : s.size(0)] = s
            return out

        batch_input_ids = pad(input_ids, self.pad_token_id)
        batch_labels = pad(labels, -100)
        batch_attn = (batch_input_ids != self.pad_token_id).long()

        pixel_values = torch.stack([torch.tensor(f["pixel_values"]) for f in features], dim=0)

        return {
            "input_ids": batch_input_ids,
            "labels": batch_labels,
            "attention_mask": batch_attn,
            "pixel_values": pixel_values,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow_remote_files", action="store_true", default=False,
                        help="Allow remote downloads from HF Hub (default: False = local-only).")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--dataset_name", type=str, default="derek-thomas/ScienceQA")
    parser.add_argument("--run_name", type=str, default="llava15_scienceqa_lora_lm_only",
                        help="Short name used to create output folder under runs/")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="Learning rate scheduler type (default: cosine).")
    parser.add_argument("--device", type=str, default="auto")


    parser.add_argument("--lora_layers",type=str, default="11111111111111111111111111111111")
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--no_merge", action="store_true", default=False,
                        help="If set, do NOT merge LoRA weights into the base model after training (default merges).")

    args = parser.parse_args()

    base_out = os.path.join("runs", args.run_name)
    adapters_out = os.path.join(base_out, "adapters")
    merged_out = os.path.join(base_out, "merged")
    os.makedirs(base_out, exist_ok=True)

    torch.manual_seed(args.seed)

    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        local_files_only=not args.allow_remote_files,
    )

    base_processor_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(base_processor_id, local_files_only=not args.allow_remote_files)
    tokenizer = processor.tokenizer
    if getattr(processor, "chat_template", None) is None and getattr(tokenizer, "chat_template", None) is not None:
        processor.chat_template = tokenizer.chat_template
    if getattr(processor, "chat_template", None) is None:
        raise ValueError(
            f"No chat_template found on processor/tokenizer loaded from {base_processor_id}. "
            "Update transformers to >=4.44 and ensure the base model repo contains a chat_template."
        )
    tokenizer.pad_token = tokenizer.eos_token

    if hasattr(model, "vision_tower") and model.vision_tower is not None:
        for p in model.vision_tower.parameters():
            p.requires_grad = False
    proj = getattr(model, "multi_modal_projector", None) or getattr(model, "mm_projector", None)
    if proj is not None:
        for p in proj.parameters():
            p.requires_grad = False

    target_modules = [
    f"model.language_model.layers.{i}.self_attn.{m}" if "proj" in m and m in ["q_proj","k_proj","v_proj","o_proj"]
    else f"model.language_model.layers.{i}.mlp.{m}"
    for i in one_hot_encoded_to_layer_index_list(args.lora_layers)
    for m in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    ]
    
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    
    model = get_peft_model(model, lora_cfg)
    
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    raw = load_dataset(args.dataset_name)
    if isinstance(raw, DatasetDict) and "train" in raw:
        ds_train = raw["train"]
    else:
        ds_train = raw["train"]

    ds_train = cast_image_column(ds_train).filter(has_image)

    def _map_fn(ex): return preprocess_example(ex, processor)
    keep_cols = ("image", "question", "choices", "answer", "hint")
    ds_train_pp = ds_train.map(_map_fn, remove_columns=[c for c in ds_train.column_names if c in keep_cols])

    collator = LLaVADataCollator(pad_token_id=tokenizer.pad_token_id)
    training_args = TrainingArguments(
        output_dir=adapters_out,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=True,
        logging_steps=10,
        save_steps=1000,
        eval_strategy="no",
        eval_steps=None,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=ds_train_pp,
    )

    trainer.train()
    trainer.save_model(adapters_out)
    processor.save_pretrained(adapters_out)
    print(f"Saved LoRA adapter + processor to {adapters_out}")

    if not args.no_merge:
        if os.path.exists(merged_out) and os.listdir(merged_out):
            raise FileExistsError(f"Merged output directory already exists and is not empty: {merged_out}")
        if hasattr(model, "language_model") and hasattr(model.language_model, "merge_and_unload"):
            merged_lm = model.language_model.merge_and_unload()
            model.language_model = merged_lm
        elif hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
        else:
            raise RuntimeError("Model does not appear to be PEFT-wrapped; cannot merge.")
        os.makedirs(merged_out, exist_ok=True)
        model.save_pretrained(merged_out)
        processor.save_pretrained(merged_out)
        print(f"Saved merged full model + processor to {merged_out}")


if __name__ == "__main__":
    main()


