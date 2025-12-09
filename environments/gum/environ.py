import re
import io
import json
import numpy as np
import verifiers as vf
from PIL import Image
from datasets import Dataset, load_dataset

import oneping
from gum import render, GumError, GumErrorType

##
## notes
##

# StarVector
# https://huggingface.co/collections/starvector/starvector-svg-datasets-svg-bench

# VCode
# https://arxiv.org/pdf/2511.02778

# starvector/text2svg-stack: has lots of diagram-like SVG images with CoG-VLM generated descriptions

# would be cool to make an ascii-art environment to test out vision based evals

##
## system prompt
##

SYSTEM_GENERATE = 'You are an assistant that writes gum.jsx code given a text description.'

##
## constants
##

RENDER_PIXELS = 512

##
## reward components
##

REWARD_BOXED = 1
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
    match = re.search(r'```([^\n]*)\n(.*?)\n```', text, re.DOTALL)
    if match:
        code = match.group(2).strip()
        return code, True
    else:
        return text, False

##
## eval functions
##

EVALUATE_PROVIDER = 'llama-cpp'
EVALUATE_SYSTEM = """Given a user request and a generated PNG image, evaluate how well the image satisfies the prompt. Use the following rubric to evaluate the image. Each of these 5 criteria is worth 1 point:

- Content is fully visible
- Relevant to the prompt
- Satisfies the prompt
- Aesthetically acceptable
- Text (if any) is legible (if no text is expected, give 1 point)

First, assess each criterion individually. Then give your final response as "SCORE: <score>", where <score> is an integer between 0 and 5."""

REGEX_SCORE = r'SCORE: (\d+)'

def eval_image(query, data, debug=False):
    # get the lvlm reply
    prompt = f'Assess the following image given the user query: {query}.'
    response = oneping.reply(prompt, image=data, system=EVALUATE_SYSTEM, provider=EVALUATE_PROVIDER)

    # print debug info
    if debug:
        print(f'[DEBUG] full eval response:\n\n{response}')

    # try to parse the score
    try:
        match = re.search(REGEX_SCORE, response)
        assert match is not None
        score = int(match.group(1))
        assert score >= 0 and score <= 5
        return score
    except Exception as e:
        print(f'[WARNING] score extraction failed: {e}')
        return 0

##
## reward functions
##

def reward_gum(prompt, text):
    reward = 0

    # extract code from text
    code, boxed = extract_code(text)

    # bonus for boxed code
    if boxed:
        reward += REWARD_BOXED

    # empty code somehow
    if len(code) == 0:
        return reward

    # parse code to svg
    data = None
    etype = None
    try:
        data = render(code, pixels=RENDER_PIXELS)
    except GumError as e:
        etype = e.error_type
    except Exception as e:
        etype = GumErrorType.UNKNOWN

    # reward based on error type
    if etype == GumErrorType.UNKNOWN:
        pass
    if etype == GumErrorType.NOCODE:
        pass
    reward += REWARD_CODE
    if etype == GumErrorType.NORETURN:
        return reward
    reward += REWARD_RETURN
    if etype == GumErrorType.NOELEMENT:
        return reward
    reward += REWARD_ELEMENT
    if etype == GumErrorType.PARSE:
        return reward
    reward += REWARD_PARSE
    if etype == GumErrorType.GENERATE:
        return reward
    reward += REWARD_GENERATE
    if etype == GumErrorType.RENDER:
        return reward
    reward += REWARD_RENDER
    if etype == GumErrorType.UNKNOWN:
        return reward

    # no data? return reward
    if data is None:
        return reward

    # convert png data to image
    try:
        image = Image.open(io.BytesIO(data))
        pixels = np.asarray(image)
        assert pixels.ndim == 3
    except Exception as e:
        print(f'[WARNING] pixel conversion failed: {e}')
        return reward

    # check if its empty
    pixel_mask = pixels.mean(axis=-1) / 255
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
    return -np.clip(frac, 0, 1)

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
        system_prompt=SYSTEM_GENERATE,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )
