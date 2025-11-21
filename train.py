import os
import torch

import bitsandbytes as bnb
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# these can all be overriden with `train` keyword arguments
default_training_args = dict(
    max_steps=500,
    save_steps=100,
    logging_steps=10,
    optim='adamw_torch',
    lr_scheduler_type='cosine',
    learning_rate=2e-4,
    warmup_ratio=0.1,
    gradient_checkpointing=True,
    gradient_accumulation_steps=1,
    bf16=True,
    weight_decay=0.001,
    dataloader_drop_last=True,
    report_to='none',
)

# pretty good params
default_lora_config = dict(
    r=128,
    lora_alpha=256,
    lora_dropout=0.1,
)

def chars_token_ratio(dataset, tokenizer, prompt_type):
    total_characters, total_tokens = 0, 0
    for example in dataset:
        user, assist = example['prompt'], example['code']
        text = prepare_sample_text(prompt_type, user, assist)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))
    return total_characters / total_tokens

def prepare_sample_text(prompt_type, user, assist=''):
    if prompt_type == 'alpaca':
        system = 'You are an assistant that writes GUM code given a text description.'
        return f'### System Prompt\n{system}\n\n### User Message\n{user}\n\n### Assistant\n{assist}'
    elif prompt_type == 'llama':
        system = 'Write GUM code to generate the following'
        return f'[INST] {system}:\n{user}\n[/INST]\n{assist}'
    else:
        raise Exception(f'Unsupported prompt type: {prompt_type}')

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

def find_all_linear_names(model, bits):
    # determine linear class
    if bits == 4:
        LinearClass = bnb.nn.Linear4bit
    elif bits == 8:
        LinearClass = bnb.nn.Linear8bitLt
    else:
        LinearClass = torch.nn.Linear

    # get linear module names
    lora_module_names = set([
        n.split('.')[-1] for n, m in model.named_modules() if isinstance(m, LinearClass)
    ])

    # needed for 16-bit
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')

    return list(lora_module_names)

def load_model_quantized(model_id, bits):
    # sort out quantization args
    if bits == 4:
        bitargs = dict(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type='nf4'
        )
    elif bits == 8:
        bitargs = dict(load_in_8bit=True)
    elif bits == 16:
        bitargs = dict(torch_dtype=torch.float16)
    else:
        raise Exception(f'Unsuppored quantization bits: {bits}')

    # actually load model
    return AutoModelForCausalLM.from_pretrained(model_id, device_map={'': 0}, **bitargs)

def train(
    data_path='data/train.jsonl', output_dir='checkpoints', model_id='codellama/CodeLlama-13b-Instruct-hf',
    bits=16, prompt_type='llama', seq_length=1024, packed=False, batch_size=8, lora_args={}, **kwargs
):
    # ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    # load dataset
    train_data = load_dataset('json', data_files=data_path, split='train')
    formatting_func = lambda b: [
        prepare_sample_text(prompt_type, u, a) for u, a in zip(b['prompt'], b['code'])
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

    # training arguments
    training_args = {**default_training_args, **kwargs}
    training_config = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        evaluation_strategy='no',
        per_device_train_batch_size=batch_size,
        **training_args,
    )

    print(f'Loading model {model_id}')
    model = load_model_quantized(model_id, bits)
    linear_modules = find_all_linear_names(model, bits)
    print(f'Linear modules: {linear_modules}')

    # hard coded for now
    lora_args1 = {
        'target_modules': linear_modules,
        **default_lora_config, **lora_args
    }
    lora_config = LoraConfig(
        bias='none',
        task_type='CAUSAL_LM',
        **lora_args1
    )

    # set up fine tuner
    train_data.start_iteration = 0
    trainer = SFTTrainer(
        model=model,
        args=training_config,
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

    # run that baby
    print('Training...')
    trainer.train()

    # save final checkpoint separately
    print('Saving last checkpoint of the model')
    final_path = os.path.join(output_dir, 'final_checkpoint')
    trainer.model.save_pretrained(final_path)
