import re
import io
import json
import numpy as np
import verifiers as vf
from PIL import Image
from datasets import Dataset, load_dataset

from .client import GumClient, GumError, ErrorType

##
## notes
##

# StarVector
# https://huggingface.co/collections/starvector/starvector-svg-datasets-svg-bench

# VCode
# https://arxiv.org/pdf/2511.02778

# starvector/text2svg-stack: has lots of diagram-like SVG images with CoG-VLM generated descriptions

# gum client
gum = GumClient()

##
## system prompt
##

SYSTEM_PROMPT = 'You are an assistant that writes gum.jsx code given a text description.'

##
## reward components
##

REWARD_CODE = 1
REWARD_PARSE = 1
REWARD_GENERATE = 1
REWARD_RETURN = 1
REWARD_ELEMENT = 1
REWARD_RENDER = 1
REWARD_MASK = 1

##
## general tools
##

def extract_code(text):
    match = re.search(r'```jsx\n(.*?)\n```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ''

##
## eval functions
##

def eval_text_image(prompt, data):
    pass

def eval_image_image(prompt, data):
    pass

##
## reward functions
##

def reward_gum(prompt, text):
    reward = 0

    # extract code from text
    code = extract_code(text)

    # parse code to svg
    try:
        data = gum.render(code)
        etype = None
    except GumError as e:
        data = None
        etype = e.error_type

    # reward based on error type
    if etype == ErrorType.UNKNOWN:
        pass
    if etype == ErrorType.NOCODE:
        pass
    reward += REWARD_CODE
    if etype == ErrorType.NORETURN:
        return reward
    reward += REWARD_RETURN
    if etype == ErrorType.NOELEMENT:
        return reward
    reward += REWARD_ELEMENT
    if etype == ErrorType.PARSE:
        return reward
    reward += REWARD_PARSE
    if etype == ErrorType.GENERATE:
        return reward
    reward += REWARD_GENERATE
    if etype == ErrorType.RENDER:
        return reward
    reward += REWARD_RENDER
    if etype == ErrorType.UNKNOWN:
        return reward

    # no data? return reward
    if data is None:
        return reward

    # convert png data to image
    image = Image.open(io.BytesIO(data))
    pixels = np.asarray(image)

    # check if its empty
    pixel_mask = pixels.max(1) / 255
    tone_std = pixel_mask.std()
    if tone_std > 0.05:
        reward += REWARD_MASK

    # use lvm critic on image
    score = eval_image(prompt, data)
    reward += score

    # return reward
    return reward

def reward_len(text, min_length=512, max_length=1024):
    frac = (len(text) - min_length) / (max_length - min_length)
    return -clamp(frac, 0, 1)

##
## dataset loading
##

def parse_docs_code(code):
    # check for format
    code = code.strip()
    if not code.startswith('\\\\'):
        print(f'Warning: Could not find format in code: {code}')
        return None

    # get head and body
    head, body = code.split('\n', maxsplit=1)
    head = head[2:].strip()
    body = body.strip()

    # return dict
    return {'prompt': head, 'answer': body}

def load_gum_dataset(data_path):
    code = json.load(data_path)
    return Dataset.from_list([parse_docs_code(line) for line in code])

##
## environment definition
##

def load_environment(
    dataset_path=None,
    use_thinking=True,
    min_length=2048,
    max_length=16384,
    **kwargs,
):
    # load dataset
    if dataset_path:
        dataset = load_gum_dataset(dataset_path)
    else:
        dataset = load_dataset('iamlemec/gum-docs', split='train')

    # thinking? parser
    ParserClass = vf.ThinkParser if use_thinking else vf.Parser
    parser = ParserClass()

    # define reward functions
    def reward_gum_function(parser, completion, answer):
        response = parser.parse_answer(completion)
        return reward_gum(response, answer)
    def reward_len_function(parser, completion, answer):
        return reward_len(completion, min_length=min_length, max_length=max_length)

    # set up reward rubric
    rubric = vf.Rubric(
        parser=parser,
        funcs=[reward_gum_function, reward_len_function, parser.get_format_reward_func()],
        weights=[1.0, 1.0, 1.0],
    )

    # set up environment
    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )
