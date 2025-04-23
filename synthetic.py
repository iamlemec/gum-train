# generate synthetic gum.js training data

##
## code generators
##

def s(x):
    return f"'{x}'"

def d(x):
    return {} if x is None else x

def eq(x, y):
    return f'let {x} = {y};'

def ret(x):
    return f'return {x};'

def obj(**entries):
    if len(entries) == 0:
        return None
    elems = ', '.join([
        (f'{k}: {v}' if k != v else k) for k, v in entries.items()
    ])
    return f'{{{elems}}}'

def wobj(**entries):
    return obj(**{
        k: (s(v) if type(v) is str else v) for k, v in entries.items()
    })

def sattr(args):
    if args is None:
        return None
    else:
        return wobj(**args)

def func(name, *args):
    args = [a for a in args if a is not None]
    elems = ', '.join(args)
    return f'{name}({elems})'

##
## text generators
##

def compress(s):
    c = re.sub(r'\s+', ' ', s).strip()
    return c if c != '' else None

def line_desc(line_attr):
    color = line_attr.get('stroke', '')
    width = line_attr.get('stroke_width', '')
    dash = line_attr.get('stroke_dasharray', '')
    return compress(f'{width} {dash} {color}')

##
## shape sampler
##

def sample_shape_code(shape, shape_attr=None, place_attr=None):
    shape_attr, place_attr = sattr(shape_attr), sattr(place_attr)
    sval = func(shape, shape_attr)
    if place_attr is None:
        return ret(sval)
    else:
        line1 = eq('shape', sval)
        line2 = ret(func('Place', 'shape', place_attr))
        return '\n'.join([line1, line2])

def sample_shape_text(shape, shape_attr=None, place_attr=None):
    ldesc = line_desc(line_attr)
    sdesc = f'A {shape}'
    if ldesc is not None:
        sdesc += ' with a {ldesc} border'

    if place_attr is None:
        return ret(sval)
    else:
        line1 = eq('shape', sval)
        line2 = ret(func('Place', 'shape', place_attr))
        return '\n'.join([line1, line2])

##
## portable options
##

twin = lambda x: (x, x)

shape_opts = [
    ('rectangle', 'Rect'),
    ('circle', 'Circle'),
    ('square', 'Square'),
    ('ellipse', 'Ellipse'),
]

color_opts = map(twin, [
    'black', 'white', 'gray', 'red', 'blue', 'green', 'yellow', 'orange', 'purple'
])

place_opts = {
}

line_opts = {
    'stroke': color_opts,
    'stroke_width': [
        ('thin', 1),
        ('medium', 2),
        ('thick', 3),
        ('heavy', 5),
    ],
    'stroke_dasharray': {
        ('dotted', 1),
        ('dashed', 3),
        ('long dashed', 5),
        ('dot dashed', [1, 3, 5, 3]),
        ('tight dashed', [4, 2]),
    }
}

##
## domain specification
##

makers = {
    'shape': sample_shape,
    'place': (place_gum, place_txt),
}
# 'layout', 'path', 'plot', 'text', 'network']
