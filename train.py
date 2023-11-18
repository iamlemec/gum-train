import os

from accelerate import Accelerator
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset

default_training_args = dict(
    max_steps=100000,
    eval_freq=1000,
    save_freq=1000,
    log_freq=100,
    batch_size=16,
    learning_rate=5e-5,
    lr_scheduler_type="linear",
    num_warmup_steps=1000,
    gradient_accumulation_steps=1,
    no_gradient_checkpointing=False,
    no_fp16=False,
    bf16=False,
    weight_decay=0.0,
)

def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    total_characters, total_tokens = 0, 0
    for example in dataset:
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def prepare_sample_text(example):
    text = f"Prompt: {example['prompt']}\n\nCode: {example['code']}"
    return text

def create_dataset(data_path, tokenizer, seq_length):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    dataset = dataset.train_test_split(test_size=0.005)
    train_data = dataset["train"]
    valid_data = dataset["test"]
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=seq_length,
        chars_per_token=chars_per_token,
    )

    return train_dataset, valid_dataset


def run_training(model_id, output_dir, train_data, val_data, **kwargs):
    print("Loading the model")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_data.start_iteration = 0

    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        run_name="llama-7b-gum",
        ddp_find_unused_parameters=False,
        **default_training_args, **kwargs
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id, load_in_8bit=True, device_map={"": Accelerator().process_index}
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=lora_config,
        packing=True,
    )

    print_trainable_parameters(trainer.model)

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    trainer.model.save_pretrained(os.path.join(output_dir, "final_checkpoint/"))

def train(data_path, output_dir, model_id, seq_length):
    os.makedirs(output_dir, exist_ok=True)
    train_dataset, eval_dataset = create_datasets(model_id, data_path, seq_length)
    run_training(model_id, output_dir, train_dataset, eval_dataset)
