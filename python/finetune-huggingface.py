from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
import json

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import  AutoTokenizer, AutoModelForImageTextToText, TrainingArguments, BitsAndBytesConfig

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
devNumber = torch.cuda.current_device() if torch.cuda.is_available() else "CPU"
print(f"Device name: {torch.cuda.get_device_name(devNumber) if torch.cuda.is_available() else 'CPU'}")
torch.cuda.empty_cache()

model_id = "Qwen/Qwen2.5-1.5B"

# 4-BIT QUANTIZATION CONFIG
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# TOKENIZER

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# PREPARE MODEL FOR QLoRA

base_model = prepare_model_for_kbit_training(base_model)

# LoRA CONFIG

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "v_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# APPLY LoRA

model = get_peft_model(base_model, lora_config)

model.print_trainable_parameters()
model.gradient_checkpointing_enable()

# LOAD DATASET

# train.jsonl format:
#
# {"instruction":"What happens if I miss onboarding training?","response":"If onboarding training is missed, your manager may follow up with reminders. Completing onboarding on time ensures access to company systems and compliance requirements."}

openhermes_ds = load_dataset("teknium/OpenHermes-2.5", split="train")
openhermes_ds = openhermes_ds.shuffle(seed=42).select(range(2000))

def convert_conversations(example):
    mapping = {
        "human": "user",
        "gpt": "assistant",
        "system": "system"
    }

    new_conversations = []

    for turn in example["conversations"]:
        new_conversations.append({
            "role": mapping.get(turn["from"], turn["from"]),
            "content": turn["value"]
        })

    return {"messages": new_conversations}

openhermes_ds = openhermes_ds.map(convert_conversations)

ultrachat_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ultrachat_ds = ultrachat_ds.shuffle(seed=42).select(range(1000))

def normalize_ultrachat(example):
    cleaned = []

    for m in example["messages"]:
        role = m.get("role")
        content = m.get("content")

        if role not in ["user", "assistant"]:
            continue

        if content is None:
            continue

        cleaned.append({
            "role": role,
            "content": str(content).strip()
        })

    return {"messages": cleaned}

ultrachat_ds = ultrachat_ds.map(normalize_ultrachat)

convolearn_ds = load_dataset("masharma/convolearn", split="train")
convolearn_ds = convolearn_ds.shuffle(seed=42).select(range(1000))

def parse_convo(text):
    lines = text.split("\n")

    messages = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("Student:"):
            messages.append(("user", line.replace("Student:", "").strip()))

        elif line.startswith("Teacher:"):
            messages.append(("assistant", line.replace("Teacher:", "").strip()))

    return messages

def convert_convolearn(example):
    convo = parse_convo(example["cleaned_conversation"])

    return {
        "messages": [
            {"role": role, "content": content}
            for role, content in convo
        ]
    }

convolearn_ds = convolearn_ds.map(convert_convolearn)


def normalize_to_ultrachat(item):
    """
    Ensures dataset matches UltraChat format:
    - messages list
    - role/content structure
    """

    msgs = item.get("messages", [])

    normalized = {"messages": []}

    for m in msgs:
        role = m.get("role")
        content = m.get("content")

        # skip invalid entries
        if role not in ["user", "assistant"] or content is None:
            continue

        normalized["messages"].append({
            "role": role,
            "content": str(content).strip()
        })

    return normalized


def convert_file(input_path, output_path):
    out = []

    with open(input_path, "r") as f:
        for line in f:
            item = json.loads(line)
            out.append(normalize_to_ultrachat(item))

    with open(output_path, "w") as f:
        for item in out:
            f.write(json.dumps(item) + "\n")


convert_file(
    "./db/onboarding_qa_520.jsonl",
    "./db/onboarding_ultrachat.jsonl"
)

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]
    
onboarding_ds = load_jsonl("./db/onboarding_ultrachat.jsonl")
onboarding_ds = Dataset.from_list(onboarding_ds)

ds = concatenate_datasets([
    openhermes_ds,
    ultrachat_ds,
    convolearn_ds,
    onboarding_ds
])

# FORMAT DATA

SYSTEM_PROMPT = """
You are an AI onboarding tutor for employees.

Your goals:
- teach clearly and patiently
- maintain a supportive professional tone
- guide users step-by-step
- reinforce company best practices
- encourage engagement
- answer concisely
- ask follow-up questions when useful

Always behave like a professional internal company trainer.
"""

def format_conversation(example):

    conversation = example["messages"]

    text = f"""
### System:
{SYSTEM_PROMPT}

### Conversation:
{conversation}
"""
    return text

# TOKENIZATION

def tokenize_function(example):

    text = format_conversation(example)

    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
    )

    tokens["labels"] = tokens["input_ids"].copy()

    return tokens

# TRAINING ARGUMENTS

training_args = TrainingArguments(
    output_dir="./qwen-2.5-1.5b-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
    fp16=True,
    optim="paged_adamw_8bit",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="none",
    completion_only_loss=True
)

# TRAINER

from transformers import Trainer
tokenized_dataset = ds.map(tokenize_function, batched=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# TRAIN

trainer.train()

# SAVE ADAPTER

model.save_pretrained("./qwen2.5-onboarding-merged")
tokenizer.save_pretrained("./qwen2.5-onboarding-merged")

print("Training complete!")