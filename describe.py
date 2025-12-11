# describe gum.js elements

import re
from gum import G

##
## constants
##

INDENT = 2

##
## utils
##

def st(x):
    if isinstance(x, str):
        return f'\"{x}\"'
    elif isinstance(x, float):
        return f'{x:.2g}'
    elif isinstance(x, (tuple, list)):
        return f'[{", ".join([st(y) for y in x])}]'
    else:
        return x

def kv(k, v):
    return k if v is True else f'{k}={{{st(v)}}}'

def kvs(props):
    return ' '.join([kv(k, v) for k, v in props.items()])

def ind(s, n=INDENT):
    pad = ' ' * n
    return '\n'.join([f'{pad}{line}' for line in s.split('\n')])

def ul(items, ind=INDENT):
    pad = ' ' * ind
    return '\n'.join([f'{pad}{item}' for item in items])

def _(object, props):
    return [object[prop] for prop in props]

##
## rect ops
##

def rect_box(x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    return x1, x2, w, h

##
## describers
##

def describe_position(x, y):
    if x < 0.3 and y < 0.3:
        return 'in the top left corner'
    elif x < 0.3 and y > 0.7:
        return 'in the bottom left corner'
    elif x > 0.7 and y < 0.3:
        return 'in the top right corner'
    elif x > 0.7 and y > 0.7:
        return 'in the bottom right corner'
    elif x < 0.3:
        return 'on the left side'
    elif x > 0.7:
        return 'on the right side'
    elif y < 0.3:
        return 'on the top side'
    elif y > 0.7:
        return 'on the bottom side'
    else:
        return 'in the center'

def describe_size(w, h):
    area = w * h
    aspect = w / h
    if area < 0.3:
        size = 'small'
    elif area < 0.7:
        size = 'medium'
    else:
        size = 'large'
    if aspect < 0.7:
        ratio = 'tall'
    elif aspect < 1.3:
        ratio = 'square'
    else:
        ratio = 'wide'
    return f'{size} and {ratio}'

def describe_proportional(s):
    if s <= 0.05:
        return 'small'
    elif s <= 0.1:
        return 'medium'
    elif s <= 0.25:
        return 'large'
    else:
        return 'huge'

def describe_lim(lim):
    l, h = lim
    return f'{l} to {h}'

def describe_func(func):
    if not isinstance(func, G.Fun):
        func = G.Fun(func)
    return str(func)

def describe_stroke_width(stroke_width):
    size = None
    if stroke_width < 1:
        size = 'thin'
    elif stroke_width == 1:
        pass
    elif stroke_width < 3:
        size = 'medium thickness'
    else:
        size = 'thick'
    if size is not None:
        return f'Give the line {size}. '
    else:
        return ''

def describe_stroke_dasharray(stroke_dasharray):
    size = None
    if stroke_dasharray == 0:
        pass
    elif stroke_dasharray < 2:
        size = 'dotted'
    elif stroke_dasharray < 5:
        size = 'dashed'
    else:
        size = 'long dashed'
    if size is not None:
        return f'Make the line {size}. '
    else:
        return ''

def describe_padding(spec):
    if spec in (None, False) or spec == 0:
        return None
    if spec is True:
        spec = 0.1
    if isinstance(spec, (float, int)):
        return describe_proportional(spec)
    elif isinstance(spec, (tuple, list)):
        if len(spec) == 2:
            h, v = map(describe_proportional, spec)
            return f'{h} horizontal and {v} vertical'
        elif len(spec) == 4:
            l, t, r, b = map(describe_proportional, spec)
            return f'{l} left, {t} top, {r} right, and {b} bottom'
        else:
            return None
    else:
        return None

def describe_padmar(padding, margin):
    dpadding = describe_padding(padding)
    dmargin = describe_padding(margin)
    if dpadding is not None and dmargin is None:
        return f'with {dpadding} padding'
    elif dmargin is not None and dpadding is None:
        return f'with {dmargin} margin'
    elif dpadding is not None and dmargin is not None:
        return f'with {dpadding} padding and {dmargin} margin'
    else:
        return ''

##
## element describers
##

def describe_children(children, indent=0):
    return ul([describe_element(c, indent=indent+INDENT) for c in children])

def describe_element(elem, indent=0):
    if not isinstance(elem, G.Element):
        desc = f'[NON-ELEMENT: {type(elem)}]'
    elif isinstance(elem, G.Rect):
        desc = 'A rectangle'
    elif isinstance(elem, G.Ellipse):
        desc = 'An ellipse'
    elif isinstance(elem, G.Circle):
        desc = 'A circle'
    elif isinstance(elem, G.Square):
        desc = 'A square'
    elif isinstance(elem, G.Frame):
        pad = elem.args.get('padding')
        mar = elem.args.get('margin')
        dchildren = describe_children(elem.children, indent=indent)
        dpadmar = describe_padmar(pad, mar)
        desc = f'A frame {dpadmar} containing:\n{dchildren}'
    elif isinstance(elem, G.Box):
        pad = elem.args.get('padding')
        mar = elem.args.get('margin')
        dchildren = describe_children(elem.children, indent=indent)
        dpadmar = describe_padmar(pad, mar)
        desc = f'A box {dpadmar} containing:\n{dchildren}'
    elif isinstance(elem, G.Box):
        dchildren = describe_children(elem.children, indent=indent)
        desc = f'A box containing:\n{dchildren}'
    elif isinstance(elem, G.HStack):
        dchildren = describe_children(elem.children, indent=indent)
        desc = f'A horizontal stack containing:\n{dchildren}'
    elif isinstance(elem, G.VStack):
        dchildren = describe_children(elem.children, indent=indent)
        desc = f'A vertical stack containing:\n{dchildren}'
    elif isinstance(elem, G.Text):
        text = elem.args.get('text')
        desc = f'The text "{text}"'
    elif isinstance(elem, G.DataPath):
        func = elem.args.get('fy')
        xlim = elem.args.get('xlim')
        dfunc = describe_func(func)
        dxlim = describe_lim(xlim)
        desc = f'A line plotting {dfunc} from {dxlim}'
    elif isinstance(elem, G.Plot):
        xlim = elem.args.get('xlim')
        ylim = elem.args.get('ylim')
        dxlim = f' with x from {describe_lim(xlim)}' if xlim is not None else ''
        dylim = f' with y from {describe_lim(ylim)}' if ylim is not None else ''
        dchildren = describe_children(elem.children, indent=indent)
        desc = f'A plot {dxlim} {dylim} of:\n{dchildren}'
    elif isinstance(elem, G.Group):
        dchildren = describe_children(elem.children, indent=indent)
        desc = f'A group containing:\n{dchildren}'
    else:
        desc = f'[UNKNOWN ELEMENT: {type(elem)}]'

    # compress non-leading spaces and return indented
    desc = re.sub(r'(?<=\S) +', ' ', desc).strip()
    return ind(desc, n=indent)
