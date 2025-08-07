# process training data

import os
import json

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
