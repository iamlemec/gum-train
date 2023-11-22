import os

from datasets import load_dataset
from accelerate import Accelerator
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset

# these can all be overriden with `train` keyword arguments
default_training_args = dict(
    max_steps=10000,
    save_steps=250,
    logging_steps=250,
    learning_rate=5e-5,
    lr_scheduler_type='linear',
    gradient_checkpointing=True,
    gradient_accumulation_steps=1,
    bf16=True,
    weight_decay=0.0,
    dataloader_drop_last=True,
    report_to='none',
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
        raise Exception(f'Unsupported prompt type: {prompt_type}')

def train(
    data_path='data/train.jsonl', output_dir='checkpoints', model_id='codellama/CodeLlama-13b-Instruct-hf',
    prompt_type='llama', seq_length=1024, packed=False, batch_size=8, **kwargs
):
    # ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    # load dataset
    train_data = load_dataset('json', data_files=data_path, split='train')
    formatting_func = lambda b: [
        prepare_sample_text(prompt_type, {'prompt': p, 'code': c})
        for p, c in zip(b['prompt'], b['code'])
    ]
    print(f'Size of training dataset: {len(train_data)}')

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    print(f'Tokens in tokenizer: {len(tokenizer)}')

    # get chars per token to size estimation
    chars_per_token = chars_token_ratio(train_data, tokenizer, prompt_type)
    print(f'Character to token ratio: {chars_per_token:.2f}')

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
        run_name=run_name,
        evaluation_strategy='no',
        per_device_train_batch_size=batch_size,
        **{**default_training_args, **kwargs},
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
        # arguments passed to dataset
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        max_seq_length=seq_length,
        infinite=True,
        packing=packed,
    )
    print_trainable_parameters(trainer.model)

    print('Training...')
    trainer.train()

    print('Saving last checkpoint of the model')
    final_path = os.path.join(output_dir, 'final_checkpoint')
    trainer.model.save_pretrained(final_path)
