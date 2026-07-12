import os
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

def main():
    # --- Step 1: Load Model & Tokenizer ---
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    max_seq_length = 2048  # Captures your max audit length (~1110 - 2313 tokens) Safely
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = True, # QLoRA 4-bit quantization
        device_map = "auto"
    )

    # --- Step 2: Configure LoRA (Conservative Baseline) ---
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,
        lora_alpha = 64,
        lora_dropout = 0.05, # Conservative dropout setting
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias = "none",
        use_gradient_checkpointing = "unsloth", # Optimized for VRAM savings
        random_state = 3407,
    )

    # --- Step 3: Load Datasets ---
    # Loading local JSONL files directly into Hugging Face Dataset format
    dataset = load_dataset("json", data_files={
        "train": "data/qwen_lora_train_v2.jsonl",
        "validation": "data/qwen_lora_validation.jsonl"
    })

    # --- Step 4: Format Samples using Chat Template ---
    # Essential step to match training layout with deployment inference structure
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return { "text" : texts }

    # Map dataset transformation
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # --- Step 5: Training Hyperparameters ---
    training_args = TrainingArguments(
        output_dir = "outputs/checkpoints/qwen_sql_lora",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        warmup_ratio = 0.03,
        weight_decay = 0.01,
        logging_steps = 1,
        evaluation_strategy = "epoch",  # Validate at the end of each epoch
        save_strategy = "epoch",        # Save checkpoints safely
        fp16 = not FastLanguageModel.is_bfloat16_supported(),
        bf16 = FastLanguageModel.is_bfloat16_supported(),
        random_seed = 3407,
    )

    # Initialize Supervised Fine-Tuning Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset = dataset["validation"],
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Set to False initially to verify exact sample boundaries
        args = training_args,
    )

    # --- Execute Run 1 ---
    trainer.train()

    # --- Step 6: Save Adapters Only ---
    # Unsloth default behavior here saves only LoRA PEFT adapters, avoiding full weights merges.
    output_dir = "outputs/checkpoints/qwen_sql_lora"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training completed successfully. Adapters saved to {output_dir}")

if __name__ == "__main__":
    main()