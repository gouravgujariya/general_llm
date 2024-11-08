import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
import data_load
import fine_tune_llm as
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


def fine_tune_llm_with_common_steps(data, a):
    # Load configuration and check GPU compatibility
    compute_dtype, bnb_config = a.load_config()
    a.check_gpu_compatibility(compute_dtype)

    # Load the base model and tokenizer
    base_model = a.load_base_model()
    tokenizer = a.LLama_tokenizer()

    # Set up PEFT configuration and training parameters
    peft_config = a.Lora_config()
    training_arguments = a.set_training_parameter()

    # Set fine-tuning parameters and train the model
    trainer = a.set_fine_tuning_parameters(base_model, data, tokenizer, peft_config, training_arguments)
    trainer.train()

    # Save the fine-tuned model
    trainer.model.save_pretrained(a.new_model)

    # Reload the model in FP16 and merge it with LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(
        a.model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=a.device_map,
    )
    model = PeftModel.from_pretrained(base_model, a.new_model)
    model = model.merge_and_unload()

    # Reload and save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer
