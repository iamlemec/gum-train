# guml

Fine-tune a code model to produce `gum.js` code.

# Training Runs

In general, 13B models are in 8-bit and 34B models are in 4-bit. Additionally, the batch size for 13B is 8, while the batch size for 34B is four. This is all with a 1024 max sequence length.

All linear layer names for Llama2 are: `['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj']`

`codellama-13b-single [r=16,alpha=32]:` The original success. Works surprisingly well! But definitely far from perfect.

`codellama-13b-r256 [r=256,alpha=512]`: Getting better with higher `r` but still not amazing.

`codellama-13b-packed [r=16,alpha=32]`: Trying out packed training. Actually has pretty good outputs but can get kind of goofy and mixed up.

`mistral-base-7b-r128`/`mistral-base-7b-r64`: Pretty coherent but way overfit in most cases. Some issues with repeating prompt.

Think about fine-tuning `mistral-7b` for code, as it is competetive with `codellama-7b` even without specialized training.

# Inference

Can help to boost `temp` and use `top_k`. Seems like `temp=1.2` and `top_k=10` works pretty well. Look at `qlora` generation params here.
