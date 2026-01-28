from typing import List, Dict, Optional
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class LocalAssistant:
    """Local LLM assistant for chat simulation."""

    def __init__(
        self,
        model_path: str,
        lora_path: Optional[str] = None,
        use_4bit: bool = False,
        device_map: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        cache_dir: str = "./model_cache",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True,
    ):
        """
        Initialize local assistant model.

        Args:
            model_path: HuggingFace model path or local path
            lora_path: Optional path to LoRA adapter weights
            use_4bit: Whether to use 4-bit quantization (QLoRA style)
            device_map: Device placement strategy
            torch_dtype: Model dtype (auto-detected if None)
            cache_dir: Directory for model cache
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling (False = greedy)
        """
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        if torch_dtype is None:
            torch_dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            )

        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            )

        logger.info(f"Loading model from {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            use_fast=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if lora_path:
            self._load_lora(lora_path)

        logger.info(
            f"Model loaded with memory footprint: "
            f"{self.model.get_memory_footprint() / (1024**3):.2f} GB"
        )

    def _load_lora(self, lora_path: str) -> None:
        """Load LoRA adapter weights."""
        from peft import PeftModel

        logger.info(f"Loading LoRA adapter from {lora_path}")
        self.model = PeftModel.from_pretrained(self.model, lora_path)

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate assistant response given conversation history.

        Args:
            messages: Conversation history as list of {"role": str, "content": str}

        Returns:
            Generated assistant message content
        """
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.do_sample else None,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        logger.debug(f"Assistant generated: {response[:100]}...")
        return response
