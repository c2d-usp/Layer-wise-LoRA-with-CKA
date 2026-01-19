# Layer-wise LoRA fine-tuning: a similarity metric approach

This repository provides the implementation of **Layer-wise LoRA fine-tuning: a similarity metric approach** to select a subset of transformer layers for fine-tuning.

## Overview

Our method identifies the most important layers to fine-tune by leveraging a representation similarity metric,  **Centered Kernel Alignment (CKA)**. Specifically, we measure the similarity between the input and output representations of each transformer layer.

We interpret high similarity as indicating a low impact to changes on internal representations, and therefore low task-specific importance.

Demonstration of our method (Comparison between a model fine-tuned with conventional LoRA and a model fine-tuned with LoRA modules only in selected layers)



https://github.com/user-attachments/assets/3198430b-1d28-4a95-9d99-06e90dc62410



## File Structure

We organize our repository by model architecture. 

- `encoder-only/`: Contains the layer selection and fine-tuning code for the encoder-only models used in this work (i.e, RoBERTa-base and DeBERTa-v3-base).
- `decoder-only/`: Contains the corresponding implementation for the decoder-only models that we use in our work (i.e, Gemma-7B, Mistral-7B-v01, and LLaMA-2-7B).
- `multimodal/`: Provides the implementation for selecting layers and fine-tuning the multimodal model LLaVA-1.5-7B.

Each folder follows the same internal structure:

- `requirements.txt`: Requirements for each architecture. We recommend creating a virtual environment for each setup. For installation, use the command: `pip install -r requirements.txt`.
- `select_layers_<architecture>.py`: Extracts internal representations and compute the CKA similarity to identify adequate subsets of transformer layers to fine-tune.
- `finetune_<architecture>.py`: Fine-tunes the selected subset of layers with LoRA.
- `scripts/`: Shell scripts (`.sh`) to fine-tune.

## Usage

For each model family, we follow the same pipeline: layer selection and fine-tune of the selected subset.

### Layer Selection

To select layers, `select_layers_architecture<>.py` extracts internal representations and measures the CKA similarity between the input and output representations of each layer. We select the top N layers with the lowest similarity, as we associate lower similarity with a higher impact on internal representations.

#### Parameters

The `select_layers_architecture<>.py` scripts share the same parameters. Except from the one specifying the task or dataset for representation extraction.
##### Shared Parameters

- `--model_name_or_path`: Path or name of the pre-trained model.
- `--number_of_samples`: Number of samples used to extract representations.
- `--batch_size`: Batch size used for inference.
- `--device`: PyTorch device name.
- `--output_dir`: Directory to save the selected layers.
- `--ratio_of_layers`: Percentage of layers to select.

##### Specific Parameter
For the encoder-only script, we use `--glue_task` to define the GLUE task we use to extract representations. For the decoder-only script, we use `--task` to define the dataset to use (either 'metamath' for MetaMathQa or 'python' for Code-Feedback). Finally, for LLaVA-1.5, we use `--dataset_name`. 

It is important to note that these scripts are designed to extract from the datasets used in our experiments.

#### Example

##### Decoder-only Models (Math or Coding)
```bash
python select_layers_encoder.py \
  --model_name_or_path meta-llama\Llama-2-7b-hf \
  --task metamath \
  --batch_size 16 \
  --device cuda \
  --number_of_samples 128 \
  --output_dir outputs/llama \
  --ratio_of_layers 0.5 \
```



### Fine-tuning

For the fine-tuning phase, each architecture type has its own `.py` script. In the /scripts/ folder, we provide one seed per configuration used in our NLG, NLU, and multimodal experiments. Importantly, in every script, the `--lora_layers` parameter receives a one-hot-encoded list (e.g, 001001001000), where zeros indicate non-selected layers and ones indicate selected layers.

### Evaluation

The evaluation of the fine-tuned models is also architecture-specific. For encoder-only models, the fine-tuning script already incorporates the evaluation step. For decoder-only models, we follow the instructions in [fxmen/pissa-dataset](https://huggingface.co/datasets/fxmeng/pissa-dataset). For the multimodal model LLaVA-1.5, we use the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) evaluation framework.

## Acknowledgements
We thank Instituto de Ciência e Tecnologia Itaú (ICTi) for the technical support, resources, and financial aid in the development of the research project. The authors would also like to thank the Programa de Bolsas Itaú (PBI) of the Centro de Ciência de Dados (C2D), supported by Itaú Unibanco S.A.
