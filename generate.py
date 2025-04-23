from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from train import prepare_sample_text, load_model_quantized

def load_checkpoint(model_id, adapter_path, bits):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = load_model_quantized(model_id, bits)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    return model, tokenizer
