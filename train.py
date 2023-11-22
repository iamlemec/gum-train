import os

from datasets import load_dataset
from accelerate import Accelerator
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset

default_training_args = dict(
    max_steps=100000,
    save_steps=250,
    learning_rate=5e-5,
    lr_scheduler_type='linear',
    gradient_checkpointing=True,
    bf16=True,
    weight_decay=0.0,
)

def chars_token_ratio(dataset, tokenizer, prompt_type):
    total_characters, total_tokens = 0, 0
    for example in dataset:
        text = prepare_sample_text(prompt_type, example)
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
        f'trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}'
    )

def prepare_sample_text(prompt_type, example):
    user, assist = example['prompt'], example['code']
    if prompt_type == 'alpaca':
        system = 'You are an assistant that produces gum.js javascript code to display images or figures given in a text description.'
        return f'### System Prompt\n{system}\n\n### User Message\n{user}\n\n### Assistant\n{assist}'
    elif prompt_type == 'llama':
        system = 'Write gum.js javascript code to display the image or figure described below'
        return f'[INST] {system}:\n{user}\n[/INST]\n{assist}'
    else:
        print(f'Unsupported prompt type: {prompt_type}')

def create_dataset(data_path, model_id, prompt_type, seq_length):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f'Tokens in tokenizer: {len(tokenizer)}')

    train_data = load_dataset('json', data_files=data_path)['train']
    print(f'Size of training dataset: {len(train_data)}')

    chars_per_token = chars_token_ratio(train_data, tokenizer, prompt_type)
    print(f'Character to token ratio: {chars_per_token:.2f}')

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=lambda s: prepare_sample_text(prompt_type, s),
        infinite=True,
        seq_length=seq_length,
        chars_per_token=chars_per_token,
    )

    return train_dataset

def run_training(model_id, output_dir, train_data, seq_length, **kwargs):
    # get run name from output_dir
    _, run_name = os.path.split(output_dir)

    # lora configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
    )

    # training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        dataloader_drop_last=True,
        evaluation_strategy='no',
        run_name=run_name,
        **default_training_args, **kwargs
    )

    print(f'Loading model {model_id}')
    model = AutoModelForCausalLM.from_pretrained(
        model_id, load_in_8bit=True, device_map={'': Accelerator().process_index}
    )

    train_data.start_iteration = 0
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        peft_config=lora_config,
        max_seq_length=seq_length,
        packing=True,
    )
    print_trainable_parameters(trainer.model)

    print('Training...')
    trainer.train()

    print('Saving last checkpoint of the model')
    trainer.model.save_pretrained(os.path.join(output_dir, 'final_checkpoint'))

def train(
    data_path='data/train.jsonl', output_dir='checkpoints', model_id='Phind/Phind-CodeLlama-34B-v2',
    prompt_type='alpaca', seq_length=4096
):
    os.makedirs(output_dir, exist_ok=True)
    train_dataset = create_dataset(data_path, model_id, prompt_type, seq_length)
    run_training(model_id, output_dir, train_dataset, seq_length)
