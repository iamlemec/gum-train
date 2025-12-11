# generate synthetic gum.js training data

import re
import random

from gum import G
from describe import describe_element

##
## code generation
##

def compress(s):
    c = re.sub(r'\s+', ' ', s).strip()
    return c if c != '' else None

def lines(*args):
    return '\n'.join(args)

def eq(x, y):
    return f'const {x} = {y}'

def ret(x):
    return f'return {x}'

##
## options and classes
##

shape_opts = [
    ('rectangle', 'Rect'),
    ('circle', 'Circle'),
    ('square', 'Square'),
    ('ellipse', 'Ellipse'),
    ('triangle', 'Triangle'),
]

frame_opts = [
    ('frame', 'Frame'),
    ('title frame', 'TitleFrame'),
]

stack_opts = [
    ('horizontal stack', 'HStack'),
    ('vertical stack', 'VStack'),
]

color_opts = [
    ('gray', 'gray'),
    ('red', 'red'),
    ('blue', 'blue'),
    ('green', 'green'),
    ('yellow', 'yellow'),
    ('orange', 'orange'),
    ('purple', 'purple'),
]

line_class = {
    'stroke': color_opts,
    'stroke-width': [
        ('medium', 2),
        ('thick', 3),
        ('heavy', 5),
    ],
    'stroke-dasharray': [
        ('dotted', 1),
        ('dashed', 3),
        ('long dashed', 5),
        ('dot dashed', [1, 3, 5, 3]),
        ('tight dashed', [4, 2]),
    ],
}

place_class = {
    'pos': [
        ('center', [0.5, 0.5]),
        ('top', [0.5, 0.8]),
        ('bottom', [0.5, 0.2]),
        ('left', [0.2, 0.5]),
        ('right', [0.8, 0.5]),
        ('top left', [0.25, 0.75]),
        ('top right', [0.75, 0.75]),
        ('bottom left', [0.25, 0.25]),
        ('bottom right', [0.75, 0.25]),
    ],
    'rad': [
        ('small', 0.1),
        ('medium', 0.3),
        ('large', 0.5),
    ],
    'rotate': [
        ('15', 15),
        ('45', 45),
        ('60', 60),
    ]
}

frame_class = {
    'padding': [
        ('small', 0.05),
        ('', True),
        ('large', 0.2),
    ],
    'margin': [
        ('small', 0.05),
        ('', True),
        ('large', 0.2),
    ],
    'border': [
        ('', True),
        ('thick', 2),
    ],
    'rounded': [
        ('small', 0.05),
        ('', True),
        ('large', 0.2),
    ],
}

##
## sample generation
## done: shape
## todo: stack, path, plot, text, network
##

def merge_dicts(*dicts):
    return {k: v for d in dicts for k, v in d.items()}

def sample_options(opts, zprob=0):
    if random.random() < zprob:
        return None
    else:
        return random.choice(opts)

def sample_classes(classes, zprob=0):
    # handle multi-class case
    if type(classes) is list:
        if len(classes) == 0:
            return {}, {}
        texts, codes = zip(*[
            sample_classes(c, zprob=zprob) for c in classes
        ])
        return (
            merge_dicts(*texts),
            merge_dicts(*codes),
        )

    # sample options
    samp0 = {
        k: sample_options(v, zprob) for k, v in classes.items()
    }
    samp = {
        k: v for k, v in samp0.items() if v is not None
    }

    # if no options are selected, return empty dicts
    if len(samp) == 0:
        return {}, {}

    # convert to text/code dicts
    text0, code0 = zip(*samp.values())
    return (
        dict(zip(samp.keys(), text0)),
        dict(zip(samp.keys(), code0)),
    )

##
## option describers
##

def line_desc(line_attr):
    color = line_attr.get('stroke', '')
    width = line_attr.get('stroke-width', '')
    dash = line_attr.get('stroke-dasharray', '')
    adjs = compress(f'{width} {dash} {color}')
    return f'Make it {adjs}. ' if adjs is not None else ''

def place_desc(place_attr):
    pos = place_attr.get('pos', None)
    rad = place_attr.get('rad', None)
    rot = place_attr.get('rotate', None)
    desc = ''
    if pos is not None:
        desc += f'Place it in the {pos}. '
    if rad is not None:
        desc += f'Make it {rad} in size. '
    if rot is not None:
        desc += f'Rotate it {rot} degrees. '
    return desc

def frame_desc(frame_attr):
    desc = ''
    padding = frame_attr.get('padding', None)
    margin = frame_attr.get('margin', None)
    border = frame_attr.get('border', None)
    rounded = frame_attr.get('rounded', None)
    if padding is not None:
        desc += f'Give it {padding} padding. '
    if margin is not None:
        desc += f'Give it {margin} margin. '
    if border is not None:
        desc += f'Give it {border} border. '
    if rounded is not None:
        desc += f'Give it {rounded} rounded. '
    return desc

##
## simple shapes
##

def sample_shape_code(shape, **attr):
    elem = Element(shape, **attr)
    return elem.render()

def sample_shape_text(shape, **attr):
    sdesc = f'Draw a {shape}. '
    ldesc = line_desc(attr)
    pdesc = place_desc(attr)
    return (sdesc + pdesc + ldesc).strip()

def sample_shape(zprob=0.5):
    stext, scode = sample_options(shape_opts)
    atext, acode = sample_classes([line_class, place_class], zprob=zprob)
    text = sample_shape_text(stext, **atext)
    code = sample_shape_code(scode, **acode)
    return text, code

##
## simple frames
##

def sample_frame_code(frame, ccode, **attr):
    elem = Element(frame, content=ccode, **attr)
    return elem.render()

def sample_frame_text(frame, ctext, **attr):
    sdesc = f'Draw a {frame}. '
    fdesc = frame_desc(attr)
    return (sdesc + fdesc + 'Give it the following content:\n' + ctext).strip()

def sample_frame(zprob=0.5):
    stext, scode = sample_options(frame_opts)
    atext, acode = sample_classes([frame_class], zprob=zprob)
    ctext, ccode = sample_shape(zprob=zprob)
    text = sample_frame_text(stext, ctext, **atext)
    code = sample_frame_code(scode, ccode, **acode)
    return text, code

##
## simple stacks
##

def sample_stack_code(stack, ccodes):
    content = '\n'.join(ccodes)
    elem = Element(stack, content=content)
    return elem.render()

def sample_stack_text(stack, ctexts):
    sdesc = f'Draw a {stack} with the following content:\n'
    return (sdesc + '\n'.join(ctexts)).strip()

def sample_stack(zprob=0.5):
    stext, scode = sample_options(stack_opts)
    snum = random.randint(2, 5)
    ctexts, ccodes = zip(*[sample_shape(zprob=zprob) for _ in range(snum)])
    text = sample_stack_text(stext, ctexts)
    code = sample_stack_code(scode, ccodes)
    return text, code
