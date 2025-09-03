#!/usr/bin/env python3.12

##
## defaults
##

MAX_SEQ_LENGTH = 2048

##
## load peft model
##

def load_peft_model(
    model_name, max_seq_length=MAX_SEQ_LENGTH, load_peft=True, lora_rank=64, gpu_memory_utilization=0.5, load_in_4bit=True, fast_inference=True, random_state=3407
):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_lora_rank=lora_rank,
        max_seq_length=max_seq_length,
        gpu_memory_utilization=gpu_memory_utilization,
        load_in_4bit=load_in_4bit,
        fast_inference=fast_inference,
    )

    if load_peft:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=[
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj',
            ],
            lora_alpha=lora_rank,
            use_gradient_checkpointing='unsloth',
            random_state=random_state,
        )

    return model, tokenizer

##
## load environment module
##

def load_env_module(env_path):
    import sys
    import importlib
    from pathlib import Path

    env_path_obj = Path(env_path)
    env_path_dir, env_path_name = str(env_path_obj.parent), str(env_path_obj.stem)

    if (reset_path := env_path_dir not in sys.path):
        sys.path.insert(0, env_path_dir)
    env_mod = importlib.import_module(env_path_name)
    if reset_path:
        sys.path.remove(env_path_dir)

    return env_mod

##
## load trainer
##

def load_trainer(
        model, tokenizer, dataset, reward_funcs, use_vllm=True, learning_rate=5e-6, adam_beta1=0.9, adam_beta2=0.99, weight_decay=0.1, warmup_ratio=0.1, lr_scheduler_type='cosine', optim='adamw_8bit', logging_steps=1, per_device_train_batch_size=1, gradient_accumulation_steps=4, num_generations=8, max_prompt_length=256, max_completion_length=1024, max_steps=250, save_steps=50, max_grad_norm=0.1, report_to='none', output_dir='outputs', **kwargs
    ):
    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        use_vllm=use_vllm, learning_rate=learning_rate, adam_beta1=adam_beta1, adam_beta2=adam_beta2, weight_decay=weight_decay, warmup_ratio=warmup_ratio, lr_scheduler_type=lr_scheduler_type, optim=optim, logging_steps=logging_steps, per_device_train_batch_size=per_device_train_batch_size, gradient_accumulation_steps=gradient_accumulation_steps, num_generations=num_generations, max_prompt_length=max_prompt_length, max_completion_length=max_completion_length, max_steps=max_steps, save_steps=save_steps, max_grad_norm=max_grad_norm, report_to=report_to, output_dir=output_dir, **kwargs
    )

    return GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = reward_funcs,
        args = training_args,
        train_dataset = dataset,
    )

##
## run model training
##

def train_model(dataset, reward_funcs, model_name='Qwen/Qwen3-4B-Thinking-2507', lora_path='grpo_saved_lora', save_path='model', save_method='merged_16bit', **kwargs):
    # load model and environment
    model, tokenizer = load_peft_model(model_name)

    # load and run trainer
    trainer = load_trainer(model, tokenizer, dataset, reward_funcs, **kwargs)
    trainer.train()

    # save lora and model
    model.save_lora(lora_path)
    model.save_pretrained_merged(save_path, tokenizer, save_method=save_method)

    # return model and tokenizer
    return model, tokenizer

##
## run inference
##

def generate_response(model, tokenizer, system_prompt, prompt, lora_path=None, temperature=0.7, top_p=0.95, max_tokens=MAX_SEQ_LENGTH, **kwargs):
    from vllm import SamplingParams

    lora_request = model.load_lora(lora_path) if lora_path else None

    text = tokenizer.apply_chat_template([
        {'role' : 'system', 'content' : system_prompt},
        {'role' : 'user', 'content' : prompt},
    ], tokenize=False, add_generation_prompt=True)

    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens, **kwargs
    )

    output = model.fast_generate(
        text,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )

    return output[0].outputs[0].text

##
## main entry point
##

HAIKU_ENV_PATH = '/home/doug/mlai/gum-prime/environments/haiku/haiku'

HAIKU_SYSTEM_PROMPT = """Haikus are special types of poems with three lines. The first line has 5 syllables, the second line has 7 syllables, and the third line has 5 syllables. Generate a haiku in response to the user's prompt.

Briefly outline what you will include in the haiku beforehand inside <think></think> tags. Then write out your haiku after that. Make sure the haiku is the last thing in your response."""

def load_haiku(env_path=HAIKU_ENV_PATH, system_prompt=HAIKU_SYSTEM_PROMPT):
    import verifiers as vf

    # load environment module
    env_mod = load_env_module(env_path)

    # construct reward functions
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

    # load dataset
    dataset = env_mod.load_haiku_dataset('train')
    dataset = dataset.map(lambda x: {
        'prompt' : [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user',   'content': x['question']},
        ],
    })

    return dataset, [reward_format_function, reward_haiku_function]

def train_haiku(model_name='Qwen/Qwen3-4B-Thinking-2507', lora_path='grpo_saved_lora', save_path='model', save_method='merged_16bit', **kwargs):
    # ignore nltk syllable warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='nltk')

    # load dataset and reward functions
    dataset, reward_funcs = load_haiku(
        env_path=HAIKU_ENV_PATH, system_prompt=HAIKU_SYSTEM_PROMPT
    )

    # run model training
    model, tokenizer = train_model(
        dataset, reward_funcs, model_name=model_name, lora_path=lora_path,
        save_path=save_path, save_method=save_method, **kwargs
    )

    # return model and tokenizer
    return model, tokenizer
