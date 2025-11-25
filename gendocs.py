# process training data

import os
import json
import llm

def load_example(file):
    # load in full text
    with open(file, 'r') as f:
        data = f.read()

    # check if multi-line
    if '\n' not in data or not data.startswith('//'):
        print(f'Warning ({file}): Could not find prompt')
        return

    # split off first comment line
    head, body = data.split('\n', maxsplit=1)
    return head[2:].strip(), body.strip()

def convert_docs(path, save=None):
    # get list of files
    files = [f for f in os.listdir(path) if f.endswith('.jsx')]

    # load in full text
    samples = [load_example(os.path.join(path, file)) for file in files]
    samples = [s for s in samples if s is not None]

    # convert to json
    data = [
        {'prompt': head, 'code': body}
        for head, body in samples
    ]

    # save or output
    if save is not None:
        with open(save, 'w') as f:
            for d in data:
                f.write(json.dumps(d) + '\n')
    else:
        return data

##
## llm image description
##

PROMPT_DESCRIBE = """Describe this image in one or two sentences. This description will be passed to another LLM that will recreate it, so write as if you were telling the LLM what to do. You can ignore size information. Include details such as theme, content, color, and style."""

def describe_image(image, model='gpt-5-mini'):
    model = llm.get_model(model)
    return model.prompt(
        PROMPT_DESCRIBE,
        attachments=[image],
    )
