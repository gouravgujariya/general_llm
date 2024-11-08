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
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


class fine_tune_llm:
  def __init__(self,Model_to_be_used_name="NousResearch/Llama-2-7b-chat-hf",new_model_name="Llama-2-7b-chat-finetune"):
    # Model and dataset
    self.model_name = Model_to_be_used_name
    self.new_model = new_model_name

    # QLoRA parameters
    self.lora_r = 64
    self.lora_alpha = 16
    self.lora_dropout = 0.1

    # bitsandbytes parameters
    self.use_4bit = True
    self.bnb_4bit_compute_dtype = "float16"
    self.bnb_4bit_quant_type = "nf4"
    self.use_nested_quant = False

    # TrainingArguments parameters
    self.output_dir = "./results"
    self.num_train_epochs = 1
    self.fp16 = False
    self.bf16 = False
    self.per_device_train_batch_size = 4
    self.per_device_eval_batch_size = 4
    self.gradient_accumulation_steps = 1
    self.gradient_checkpointing = True
    self.max_grad_norm = 0.3
    self.learning_rate = 2e-4
    self.weight_decay = 0.001
    self.optim = "paged_adamw_32bit"
    self.lr_scheduler_type = "cosine"
    self.max_steps = -1
    self.warmup_ratio = 0.03
    self.group_by_length = True
    self.save_steps = 0
    self.logging_steps = 25
    self.max_seq_length = None
    self.packing = False
    self.device_map = {"": 0}

  def load_config(self):
    compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=self.use_4bit,
        bnb_4bit_quant_type=self.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=self.use_nested_quant,
    )
    return compute_dtype,bnb_config

  def check_gpu_compatibility(self,compute_dtype):
    if compute_dtype == torch.float16 and self.use_4bit:
      major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

  def load_base_model(self,bnb_config):
    base_model = AutoModelForCausalLM.from_pretrained(
        self.model_name,
        quantization_config=bnb_config,
        device_map=self.device_map)
    self.model.config.use_cache = False
    self.model.config.pretraining_tp = 1
    return base_model

  def LLama_tokenizer(self):
    tokenizer = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

  def Lora_config(self):
    peft_config = LoraConfig(
        lora_alpha=self.lora_alpha,
        lora_dropout=self.lora,
        r=self.lora_r,
        bias="none",
        task_type="CAUSAL_LM")
    return peft_config

  def set_training_parameter(self):
    training_arguments = TrainingArguments(
      output_dir=self.output_dir,
      num_train_epochs=self.num_train_epochs,
      per_device_train_batch_size=self.per_device_train_batch_size,
      gradient_accumulation_steps=self.gradient_accumulation_steps,
      optim=self.optim,
      save_steps=self.save_steps,
      logging_steps=self.logging_steps,
      learning_rate=self.learning_rate,
      weight_decay=self.weight_decay,
      fp16=self.fp16,
      bf16=self.bf16,
      max_grad_norm=self.max_grad_norm,
      max_steps=self.max_steps,
      warmup_ratio=self.warmup_ratio,
      group_by_length=self.group_by_length,
      lr_scheduler_type=self.lr_scheduler_type,
      report_to="tensorboard"
      )
    return training_arguments

  def set_fine_tuning_parameters(self,model,dataset,tokenizer,peft_config,training_arguments):
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False
    )
    return trainer

  def model_save(self,Trainer):
    return Trainer.model.save_pretrained(self.new_model);

