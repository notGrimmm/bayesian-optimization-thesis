from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

from trl import SFTTrainer

from datasets import (
    load_dataset,
    Dataset,
    concatenate_datasets
)

from trl import SFTTrainer, SFTConfig

import torch
import json

# =========================================================
# LOAD BEST PARAMETERS
# =========================================================

best_params = {
    "learning_rate": 0.0002960652557964277,
    "lora_r": 12,
    "lora_dropout": 0.0672606157822065
}

# =========================================================
# DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# =========================================================
# MODEL
# =========================================================

model_id = "Qwen/Qwen2.5-1.5B"

# OPTIONAL QLORA CONFIG
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# =========================================================
# TOKENIZER
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token

# =========================================================
# MODEL LOAD
# =========================================================

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)

# =========================================================
# APPLY BEST LORA CONFIG
# =========================================================

peft_config = LoraConfig(
    r=best_params["lora_r"],
    lora_alpha=16,
    lora_dropout=best_params["lora_dropout"],
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
)

model = get_peft_model(model, peft_config)

model.print_trainable_parameters()

# =========================================================
# LOAD DATASETS
# =========================================================

# OpenHermes
openhermes_ds = load_dataset(
    "teknium/OpenHermes-2.5",
    split="train"
).shuffle(seed=42).select(range(1000))

def convert_conversations(example):

    mapping = {
        "human": "user",
        "gpt": "assistant",
        "system": "system"
    }

    messages = []

    for turn in example["conversations"]:
        messages.append({
            "role": mapping.get(turn["from"], turn["from"]),
            "content": turn["value"]
        })

    return {"messages": messages}

openhermes_ds = openhermes_ds.map(convert_conversations)

# UltraChat
ultrachat_ds = load_dataset(
    "HuggingFaceH4/ultrachat_200k",
    split="train_sft"
).shuffle(seed=42).select(range(500))

def normalize_ultrachat(example):

    cleaned = []

    for m in example["messages"]:

        if m["role"] not in ["user", "assistant"]:
            continue

        cleaned.append({
            "role": m["role"],
            "content": str(m["content"]).strip()
        })

    return {"messages": cleaned}

ultrachat_ds = ultrachat_ds.map(normalize_ultrachat)

# =========================================================
# CUSTOM ONBOARDING DATA
# =========================================================

def load_jsonl(path):

    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

onboarding_ds = load_jsonl("./db/onboarding_ultrachat.jsonl")

onboarding_ds = Dataset.from_list(onboarding_ds)

# =========================================================
# MERGE DATASETS
# =========================================================

dataset = concatenate_datasets([
    openhermes_ds,
    ultrachat_ds,
    onboarding_ds
])

dataset = dataset.shuffle(seed=42)

split = dataset.train_test_split(test_size=0.1)

train_dataset = split["train"]
eval_dataset = split["test"]

def to_prompt_completion(example):
    messages = example["messages"]

    # Keep only valid roles/content
    messages = [
        {
            "role": m["role"],
            "content": str(m["content"]).strip()
        }
        for m in messages
        if m.get("role") in ["system", "user", "assistant"]
        and m.get("content")
    ]

    # Need final message to be assistant
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        return {
            "prompt": None,
            "completion": None
        }

    prompt_messages = messages[:-1]
    completion_message = messages[-1]
    # =========================================================
    # SYSTEM PROMPT
    # =========================================================

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

    # Ensure system prompt exists
    if not any(m["role"] == "system" for m in prompt_messages):
        prompt_messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.strip()
            }
        ] + prompt_messages

    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    completion = (
        completion_message["content"].strip()
        + tokenizer.eos_token
    )

    return {
        "prompt": prompt,
        "completion": completion
    }

train_dataset = train_dataset.map(
    to_prompt_completion,
    remove_columns=train_dataset.column_names
)

eval_dataset = eval_dataset.map(
    to_prompt_completion,
    remove_columns=eval_dataset.column_names
)


# =========================================================
# TRAINING ARGS
# =========================================================

training_args = SFTConfig(
    output_dir="./qwen-onboarding-final-v2",

    learning_rate=best_params["learning_rate"],

    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    num_train_epochs=3,

    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",

    fp16=True,
    bf16=False,

    optim="paged_adamw_8bit",

    lr_scheduler_type="cosine",
    warmup_ratio=0.05,

    max_length=512,

    # Important
    completion_only_loss=True,

    report_to="none"
)

# =========================================================
# TRAINER
# =========================================================

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

# =========================================================
# TRAIN
# =========================================================
for name, param in model.named_parameters():
    if param.requires_grad:
        param.data = param.data.float()

trainer.train()

# =========================================================
# SAVE MODEL
# =========================================================

trainer.model.save_pretrained("./qwen-onboarding-final-v2")

tokenizer.save_pretrained("./qwen-onboarding-final-v2")

print("Final fine-tuning complete.")