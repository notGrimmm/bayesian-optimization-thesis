import os
import gc
import json
import logging
from datetime import datetime

import optuna
import torch

from datasets import (
    Dataset,
    concatenate_datasets,
    load_dataset
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from trl import (
    SFTTrainer,
    SFTConfig
)

# =========================================================
# LOGGER
# =========================================================

os.makedirs("logs", exist_ok=True)

log_filename = (
    f"logs/bayesian_"
    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

logger.info("Logger initialized")

# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

logger.info(f"Using device: {device}")

if torch.cuda.is_available():

    logger.info(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

torch.cuda.empty_cache()

# =========================================================
# MODEL
# =========================================================

model_id = "Qwen/Qwen2.5-1.5B"

# =========================================================
# TOKENIZER
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token


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

# =========================================================
# OPENHERMES
# =========================================================

logger.info("Loading OpenHermes")

openhermes_ds = load_dataset(
    "teknium/OpenHermes-2.5",
    split="train"
)

openhermes_ds = (
    openhermes_ds
    .shuffle(seed=42)
    .select(range(2000))
)

def convert_conversations(example):

    mapping = {
        "human": "user",
        "gpt": "assistant",
        "system": "system"
    }

    new_conversations = []

    for turn in example["conversations"]:

        new_conversations.append({
            "role": mapping.get(
                turn["from"],
                turn["from"]
            ),
            "content": turn["value"]
        })

    return {
        "messages": new_conversations
    }

openhermes_ds = openhermes_ds.map(
    convert_conversations
)

# =========================================================
# ULTRACHAT
# =========================================================

logger.info("Loading UltraChat")

ultrachat_ds = load_dataset(
    "HuggingFaceH4/ultrachat_200k",
    split="train_sft"
)

ultrachat_ds = (
    ultrachat_ds
    .shuffle(seed=42)
    .select(range(1000))
)

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

    return {
        "messages": cleaned
    }

ultrachat_ds = ultrachat_ds.map(
    normalize_ultrachat
)

# =========================================================
# CONVOLEARN
# =========================================================

logger.info("Loading ConvoLearn")

convolearn_ds = load_dataset(
    "masharma/convolearn",
    split="train"
)

convolearn_ds = (
    convolearn_ds
    .shuffle(seed=42)
    .select(range(500))
)

def parse_convo(text):

    lines = text.split("\n")

    messages = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if line.startswith("Student:"):

            messages.append((
                "user",
                line.replace("Student:", "").strip()
            ))

        elif line.startswith("Teacher:"):

            messages.append((
                "assistant",
                line.replace("Teacher:", "").strip()
            ))

    return messages

def convert_convolearn(example):

    convo = parse_convo(
        example["cleaned_conversation"]
    )

    return {
        "messages": [
            {
                "role": role,
                "content": content
            }
            for role, content in convo
        ]
    }

convolearn_ds = convolearn_ds.map(
    convert_convolearn
)

# =========================================================
# CUSTOM DATASET
# =========================================================

logger.info("Loading onboarding dataset")

def load_jsonl(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return [
            json.loads(line)
            for line in f
        ]

onboarding_ds = load_jsonl(
    "./db/onboarding_ultrachat.jsonl"
)

onboarding_ds = Dataset.from_list(
    onboarding_ds
)

# =========================================================
# MERGE DATASETS
# =========================================================

logger.info("Merging datasets")

dataset = concatenate_datasets([
    openhermes_ds,
    ultrachat_ds,
    convolearn_ds,
    onboarding_ds
])

dataset = dataset.shuffle(seed=42)

split = dataset.train_test_split(
    test_size=0.1
)

train_dataset = split["train"]
eval_dataset = split["test"]

logger.info(
    f"Train Dataset Size: {len(train_dataset)}"
)

logger.info(
    f"Eval Dataset Size: {len(eval_dataset)}"
)

# =========================================================
# PROMPT / COMPLETION FORMAT
# =========================================================

def to_prompt_completion(example):

    messages = example["messages"]

    messages = [
        {
            "role": m["role"],
            "content": str(
                m["content"]
            ).strip()
        }
        for m in messages
        if m.get("role")
        in ["system", "user", "assistant"]
        and m.get("content")
    ]

    if (
        len(messages) < 2
        or messages[-1]["role"] != "assistant"
    ):

        return {
            "prompt": None,
            "completion": None
        }

    prompt_messages = messages[:-1]

    if not any(
        m["role"] == "system"
        for m in prompt_messages
    ):

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
        messages[-1]["content"].strip()
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
# REMOVE INVALID ROWS
# =========================================================

train_dataset = train_dataset.filter(
    lambda x:
        x["prompt"] is not None
        and x["completion"] is not None
)

eval_dataset = eval_dataset.filter(
    lambda x:
        x["prompt"] is not None
        and x["completion"] is not None
)

logger.info(
    f"Filtered Train Dataset: {len(train_dataset)}"
)

logger.info(
    f"Filtered Eval Dataset: {len(eval_dataset)}"
)

# =========================================================
# OBJECTIVE FUNCTION
# =========================================================

def objective(trial):

    model = None
    trainer = None

    try:

        logger.info(
            f"Starting Trial {trial.number}"
        )

        learning_rate = trial.suggest_float(
            "learning_rate",
            1e-5,
            5e-4,
            log=True
        )

        lora_r = trial.suggest_categorical(
            "lora_r",
            [4, 8, 12]
        )

        lora_dropout = trial.suggest_float(
            "lora_dropout",
            0.01,
            0.15
        )

        logger.info(
            f"Trial {trial.number} Params | "
            f"lr={learning_rate} | "
            f"lora_r={lora_r} | "
            f"dropout={lora_dropout}"
        )

        # =====================================================
        # MODEL LOAD
        # =====================================================
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    trust_remote_code=True,
                    dtype=torch.float16,
                    quantization_config=bnb_config,
                )

        model = prepare_model_for_kbit_training(
                    model
                )
        # =====================================================
        # LORA
        # =====================================================

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=16,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "v_proj",
            ]
        )

        model = get_peft_model(
            model,
            peft_config
        )

        model.print_trainable_parameters()

        # =====================================================
        # TRAINING CONFIG
        # =====================================================

        training_args = SFTConfig(
            output_dir="./tmp",

            learning_rate=learning_rate,

            per_device_train_batch_size=1,

            gradient_accumulation_steps=8,

            num_train_epochs=1,

            logging_steps=10,

            save_strategy="no",

            eval_strategy="epoch",

            fp16=True,

            bf16=False,

            optim="paged_adamw_8bit",

            lr_scheduler_type="cosine",

            warmup_ratio=0.05,

            max_length=256,

            completion_only_loss=True,

            report_to="none"
        )

        # =====================================================
        # TRAINER
        # =====================================================

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )

        # =====================================================
        # TRAIN
        # =====================================================

        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        trainer.train()

        # =====================================================
        # EVALUATE
        # =====================================================

        eval_results = trainer.evaluate()

        eval_loss = eval_results["eval_loss"]

        logger.info(
            f"Trial {trial.number} Complete | "
            f"Eval Loss: {eval_loss}"
        )

        return eval_loss

    except Exception as e:

        logger.exception(
            f"Trial {trial.number} failed: {str(e)}"
        )

        raise

    finally:

        logger.info(
            f"Cleaning memory for "
            f"trial {trial.number}"
        )

        if trainer is not None:
            del trainer

        if model is not None:
            del model

        gc.collect()

        torch.cuda.empty_cache()

        torch.cuda.ipc_collect()

# =========================================================
# OPTUNA STUDY
# =========================================================

logger.info("Starting Bayesian Optimization")

study = optuna.create_study(
    direction="minimize"
)

study.optimize(
    objective,
    n_trials=3
)

# =========================================================
# RESULTS
# =========================================================

logger.info(
    f"Best Params: {study.best_params}"
)

logger.info(
    f"Best Value: {study.best_value}"
)

print("\nBest Parameters:")
print(study.best_params)

print("\nBest Eval Loss:")
print(study.best_value)