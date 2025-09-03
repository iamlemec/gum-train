#!/usr/bin/env python
# coding: utf-8

# ## Unsloth

# In[1]:


import sys
import importlib
from pathlib import Path


# In[2]:


import torch


# ### Load Model

# Load up `Qwen3-4B-Thinking-2507`, and set parameters

# In[3]:


max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower


# In[4]:


from unsloth import FastLanguageModel

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


# ### Environment Spec

# In[5]:


env_path = Path('/home/doug/mlai/gum-prime/environments/haiku/haiku')
env_path_dir, env_path_name = str(env_path.parent), str(env_path.stem)
if env_path_dir not in sys.path:
    sys.path.insert(0, env_path_dir)
env_mod = importlib.import_module(env_path_name)


# In[19]:


format_prompt = f'Briefly outline what you will include in the haiku beforehand inside <think></think> tags. Then write out your haiku after that. Make sure the haiku is the last thing in your response.'
system_prompt = f'{env_mod.SYSTEM_PROMPT}\n\n{format_prompt}'
print(system_prompt)


# In[14]:


import verifiers as vf
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


# In[15]:


reward_format_function([[
    {'role': 'assistant', 'content': '<think>hmm</think>hello world'}
]])


# In[16]:


reward_haiku_function([[{'content': f"""<think>hmm</think>
hello world today
this is a test of the sun
what shall we do now
"""}]])


# ### Load Dataset

# In[17]:


dataset = env_mod.load_haiku_dataset('train')
dataset = dataset.map(lambda x: {
    "prompt" : [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": x["question"]},
    ],
})
dataset[0]


# <a name="Train"></a>
# ### Train the model
# 
# Now set up GRPO Trainer and all configurations!

# In[11]:


from trl import GRPOConfig, GRPOTrainer


# In[12]:


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


# And let's run the trainer! If you scroll up, you'll see a table of rewards. The goal is to see the `reward` column increase!
# 
# You might have to wait 150 to 200 steps for any action. You'll probably get 0 reward for the first 100 steps. Please be patient!
# 
# | Step | Training Loss | reward    | reward_std | completion_length | kl       |
# |------|---------------|-----------|------------|-------------------|----------|
# | 1    | 0.000000      | 0.125000  | 0.000000   | 200.000000        | 0.000000 |
# | 2    | 0.000000      | 0.072375  | 0.248112   | 200.000000        | 0.000000 |
# | 3    | 0.000000      | -0.079000 | 0.163776   | 182.500000        | 0.000005 |
# 

# In[13]:


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


# <a name="Inference"></a>
# ### Inference
# Now let's try the model we just trained! First, let's first try the model without any GRPO trained:

# In[22]:


text = tokenizer.apply_chat_template([
    {"role" : "system", "content" : system_prompt},
    {"role" : "user", "content" : "Write a haiku about the old forest."},
], tokenize = False, add_generation_prompt = True)

from vllm import SamplingParams
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


# And now with the LoRA we just trained with GRPO - we first save the LoRA first!

# In[14]:


model.save_lora("grpo_saved_lora")


# Now we load the LoRA and test:

# In[18]:


text = tokenizer.apply_chat_template([
    {"role" : "system", "content" : system_prompt},
    {"role" : "user", "content" : "Write a haiku about the old forest."},
], tokenize = False, add_generation_prompt = True)

from vllm import SamplingParams
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


# Our reasoning model is much better - it's not always correct, since we only trained it for an hour or so - it'll be better if we extend the sequence length and train for longer!

# <a name="Save"></a>
# ### Saving to float16 for VLLM
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.

# In[23]:


# Merge to 16bit
model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# Merge to 4bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# Just LoRA adapters
if False:
    model.save_pretrained("model")
    tokenizer.save_pretrained("model")
if False:
    model.push_to_hub("hf/model", token = "")
    tokenizer.push_to_hub("hf/model", token = "")


# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.
# 
# Some supported quant methods (full list on our [Wiki page](https://github.com/unslothai/unsloth/wiki#gguf-quantization-options)):
# * `q8_0` - Fast conversion. High resource use, but generally acceptable.
# * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
# * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.
# 
# [**NEW**] To finetune and auto export to Ollama, try our [Ollama notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)

# In[22]:


# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method='f32')
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your user?name!
if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

# Save to multiple GGUF options - much faster if you want multiple!
if False:
    model.push_to_hub_gguf(
        "hf/model", # Change hf to your username!
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = "",
    )


# Now, use the `model-unsloth.gguf` file or `model-unsloth-Q4_K_M.gguf` file in llama.cpp or a UI based system like Jan or Open WebUI. You can install Jan [here](https://github.com/janhq/jan) and Open WebUI [here](https://github.com/open-webui/open-webui)
# 
# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
# 
# Some other links:
# 1. Train your own reasoning model - Llama GRPO notebook [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
# 2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
# 3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
# 6. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks)!
# 
# <div class="align-center">
#   <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
#   <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
#   <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>
# 
#   Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
# </div>
# 
