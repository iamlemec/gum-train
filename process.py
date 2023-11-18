# process training data

import re
import json

def parse_header(head):
    return dict(
        re.split(r'(?<!\\)\=', i) for i in re.split(r'(?<!\\)\|', head)
    )

def unescape_text(text):
    return re.sub(r'(?<!\\)\\([\=\|\*_])', r'\1', text)

def convert_examples(path, save=None):
    # load in full text
    with open(path, 'r') as f:
        data = f.read()

    # split into cells
    _, *cells = re.split(r'\n\n+', data)
    heads, bodys = zip(*[c.split('\n', maxsplit=1) for c in cells])
    heads, = zip(*[re.match(r'^\!gum\* \[([^\]]+)\]$', h).groups() for h in heads])

    # parse header
    infos = [parse_header(h) for h in heads]
    caps = [i['caption'] for i in infos]

    # merge data
    data = list(zip(caps, bodys))

    # save or output
    if save is not None:
        with open(save, 'w') as f:
            for c, b in data:
                line = json.dumps({'prompt': c, 'code': b})
                f.write(f'{line}\n')
    else:
        return data
