import re
import io
import verifiers as vf
from PIL import Image

from client import GumClient, GumError, ErrorType

##
## general tools
##

def extract_code(text):
    match = re.search(r'```jsx\n(.*?)\n```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ''

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
## gum client
##

gum = GumClient()

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

    # check if its empty
    pixel_mask = image.max(1) / 255
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
## environment definition
##

def load_environment(
    use_thinking=True,
    min_length=2048,
    max_length=16384,
):
    # define reward functions
    def reward_gum_function(parser, completion, **kwargs):
        reply = parser.parse_answer(completion)
        return reward_gum(reply)
    def reward_len_function(parser, completion, **kwargs):
        reply = parser.parse_answer(completion)
        return reward_len(reply, min_length=min_length, max_length=max_length)

    # load training data
    dataset = load_haiku_dataset('train')
    train_dataset = dataset.select(range(num_train_examples))
    eval_dataset = dataset.select(
        range(num_train_examples, num_train_examples + num_eval_examples)
    )

    # thinking? parser
    ParserClass = vf.ThinkParser if think else vf.Parser
    parser = ParserClass()

    # set up haiku reward rubric
    rubric = vf.Rubric(
        parser=parser,
        funcs=[
            reward_gum_function,
            reward_len_function,
        ],
        weights=[1.0, 1.0, 1.0],
    )

    # set up environment
    return vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
    )
