#!/usr/bin/env python3.12

##
## Unsloth
##

import sys
import importlib
from pathlib import Path

from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
import verifiers as vf

##
## Load Model
##

max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen3-4B-Thinking-2507",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

##
## Environment Spec
##

# load environment
env_path = Path('/home/doug/mlai/gum-prime/environments/haiku/haiku')
env_path_dir, env_path_name = str(env_path.parent), str(env_path.stem)
if env_path_dir not in sys.path:
    sys.path.insert(0, env_path_dir)
env_mod = importlib.import_module(env_path_name)

# system prompt
format_prompt = f'Briefly outline what you will include in the haiku beforehand inside <think></think> tags. Then write out your haiku after that. Make sure the haiku is the last thing in your response.'
system_prompt = f'{env_mod.SYSTEM_PROMPT}\n\n{format_prompt}'

# reward functions
parser = vf.ThinkParser(extract_fn=env_mod.parse_haiku)
reward_format_fn = parser.get_format_reward_func()
def get_responses(completions):
    return [
        completion[0]['content'] for completion in completions
    ]
def reward_format_function(completions, **kwargs):
    responses = get_responses(completions)
    return [
        reward_format_fn([
            {'role': 'assistant', 'content': content}
            for content in responses
        ])
    ]
def reward_haiku_function(completions, **kwargs):
    responses = get_responses(completions)
    return [
        env_mod.reward_haiku_function(parser, content)
        for content in responses
    ]

##
## Load Dataset
##

dataset = env_mod.load_haiku_dataset('train')
dataset = dataset.map(lambda x: {
    "prompt" : [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": x["question"]},
    ],
})

##
## Train the model
##

# training args
training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 8, # Decrease if out of memory
    max_prompt_length = 256,
    max_completion_length = 1024,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 250,
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
)

# run trainer
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        reward_format_function,
        reward_haiku_function,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()

# save lora
model.save_lora("grpo_saved_lora")

##
## Base Inference
##

if False:
    text = tokenizer.apply_chat_template([
        {"role" : "system", "content" : system_prompt},
        {"role" : "user", "content" : "Write a haiku about the old forest."},
    ], tokenize = False, add_generation_prompt = True)

    sampling_params = SamplingParams(
        temperature = 0.8,
        top_p = 0.95,
        max_tokens = 1024,
    )
    output = model.fast_generate(
        [text],
        sampling_params = sampling_params,
        lora_request = None,
    )[0].outputs[0].text

    print(output)

##
## Trained Inference
##

if False:
    text = tokenizer.apply_chat_template([
        {"role" : "system", "content" : system_prompt},
        {"role" : "user", "content" : "Write a haiku about the old forest."},
    ], tokenize = False, add_generation_prompt = True)

    sampling_params = SamplingParams(
        temperature = 0.8,
        top_p = 0.95,
        max_tokens = 1024,
    )
    output = model.fast_generate(
        text,
        sampling_params = sampling_params,
        lora_request = model.load_lora("grpo_saved_lora"),
    )[0].outputs[0].text

    print(output)

##
## Save Model
##

# Merge to 16bit
model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)

# Merge to 4bit
if False:
    model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)

# Just LoRA adapters
if False:
    model.save_pretrained("model")
    tokenizer.save_pretrained("model")

##
## Save Model to GGUF
##

# Save to f32 GGUF
if False:
    model.save_pretrained_gguf("model", tokenizer, quantization_method='f32')

# Save to b16 GGUF
if False:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="bf16")

# Save to q5_k_m GGUF
if False:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="q5_k_m")
