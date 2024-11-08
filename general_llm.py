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
import fine_tune_llm
import train_save_model
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

from fastapi import FastAPI, File, UploadFile
import json

app = FastAPI()

@app.post("/fine-tune/")
async def fine_tune_model(data_format: str, file: UploadFile = File(...)):
    a = fine_tune_llm()  # Assuming you have an instance of your class
    data_path = file.file.name

    # Load data based on the provided format
    if data_format == "raw":
        data = load_raw_data(data_path)
    elif data_format == "qa":
        data = load_question_answer(data_path)
    else:
        return {"error": "Unsupported data format"}

    # Call the common fine-tuning function
    model, tokenizer = train_save_model(data, a)

    return {"status": "Model fine-tuned successfully"}
