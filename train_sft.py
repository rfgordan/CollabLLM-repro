# general python imports
from typing import Any
import logging

# HF / TRL / PyTorch stack
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import load_dataset
import torch

# project code
from dataset_utils import multiturn_dataset_to_sft

logger = logging.getLogger(__name__)

def load_and_train_sft(
        hf_model_path: str, 
        hf_dataset_path: str,
        learning_rate: float = 2e-5,
        batch_size: int = 4,):
    
    """ Load a Hugging Face model and perform supervised fine-tuning (SFT) on the provided dataset. """

    # logger.info(f"Running SFT on model: {hf_model_path} with dataset: {hf_dataset_path}")
    # logger.info(f"cuda current device: f{torch.cuda.current_device()}")

    # # use bf16 if supported by GPU
    # dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    dtype = torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=dtype,
        bnb_4bit_quant_storage=dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        cache_dir="./model_cache",
        quantization_config=bnb_config,
        torch_dtype=dtype,
        device_map={"":0}
    )

    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_path,
        cache_dir="./model_cache",
        use_fast=True,
    )

    logger.info(f"Model {hf_model_path} loaded with memory footprint: {model.get_memory_footprint()/(1024**3):.2f} GB")

    dataset = load_dataset(hf_dataset_path, cache_dir="./data_cache")
    dataset_clean = multiturn_dataset_to_sft(dataset, eval_ratio=0.1, lower_bound_metric=0.1)

    logger.info(f"Dataset {hf_dataset_path} loaded and preprocessed.\n {dataset_clean}")

    # LORA setup
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # training
    training_arguments = SFTConfig(
        output_dir="./results",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        logging_steps=True,
        # max_seq_length=1024,
        learning_rate=learning_rate,
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_clean["train"],
        eval_dataset=dataset_clean["eval"],
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_arguments,
    )
    
    trainer.train()

    # custom eval on data?

    # save trained model to hf?