## PLAN: recreate gum.js Element classes and have code AND text rendering methods
## additionally have a classmethod for random generation (sample)
## this is the way

##
## constants
##

INDENT = 2

##
## utils
##

def st(x):
    if type(x) is str:
        return f"\"{x}\""
    elif type(x) is float:
        return f"{x:.2g}"
    elif type(x) is list:
        return '[' + ', '.join([st(y) for y in x]) + ']'
    else:
        return x

def kv(k, v):
    return k if v is True else f'{k}={{{st(v)}}}'

def kvs(props):
    return ' '.join([kv(k, v) for k, v in props.items()])

def ind(s, n=INDENT):
    pad = ' ' * n
    return '\n'.join([f'{pad}{line}' for line in s.split('\n')])

def _(object, props):
    return [object[prop] for prop in props]

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

##
## geometry
##

def size_rect(x, y, w, h):
    return [x, y, x + w, y + h]

def rad_rect(x, y, rw, rh):
    return [x - rw, y - rh, x + rw, y + rh]

def rect_size(x1, y1, x2, y2):
    return [x1, y1, x2 - x1, y2 - y1]

##
## classes
##

class Element:
    def __init__(self, tag='Element', unary=True, **props):
        self.tag = tag
        self.unary = unary
        self.props = props

        # convert to spec
        if 'pos' in props and 'rad' in props:
            x, y = props.pop('pos')
            rx, ry = props.pop('rad')
            props['rect'] = rad_rect(x, y, rx, ry)

    def inner_jsx(self):
        return ''

    def render_jsx(self):
        props = kvs(self.props)
        pad = ' ' if len(props) > 0 else ''
        if self.unary:
            return f'<{self.tag} {props}{pad}/>'
        else:
            return f'<{self.tag} {props}>{self.inner_jsx()}</{self.tag}>'

    def render_text(self):
        raise NotImplementedError(f'render_text not implemented for this {self.tag}')

class Group(Element):
    def __init__(self, *children, **props):
        super().__init__('Group', unary=False, **props)
        self.children = children

    def inner_jsx(self):
        cstr = '\n'.join([ind(c.render_jsx()) for c in self.children])
        return f'\n{cstr}\n' if len(cstr) > 0 else ''

    def inner_text(self, prefix=''):
        return '\n'.join([ind(prefix + c.render_text()) for c in self.children])

    def render_text(self):
        return f'A group containing:\n{self.inner_text(prefix="- ")}'

class Rect(Element):
    def __init__(self, **props):
        super().__init__('Rect', **props)

    def render_text(self):
        x1, y1, x2, y2 = self.props['rect']
        x, y, w, h = rect_size(x1, y1, x2, y2)
        return f'A {describe_size(w, h)} rectangle that is {describe_position(x, y)}'

class Ellipse(Element):
    def __init__(self, **props):
        super().__init__('Ellipse', **props)

    def render_text(self):
        x1, y1, x2, y2 = self.props['rect']
        x, y, w, h = rect_size(x1, y1, x2, y2)
        return f'An {describe_size(w, h)} ellipse that is {describe_position(x, y)}'

class Circle(Element):
    def __init__(self, **props):
        super().__init__('Circle', **props)

    def render_text(self):
        x1, y1, x2, y2 = self.props['rect']
        x, y, w, h = rect_size(x1, y1, x2, y2)
        s = min(w, h)
        return f'A {describe_size(s, s)} circle that is {describe_position(x, y)}'

class Square(Element):
    def __init__(self, **props):
        super().__init__('Square', **props)

    def render_text(self):
        x1, y1, x2, y2 = self.props['rect']
        x, y, w, h = rect_size(x1, y1, x2, y2)
        s = min(w, h)
        return f'A {describe_size(s, s)} square that is {describe_position(x, y)}'

class Triangle(Element):
    def __init__(self, **props):
        super().__init__('Triangle', **props)

    def render_text(self):
        x1, y1, x2, y2 = self.props['rect']
        x, y, w, h = rect_size(x1, y1, x2, y2)
        return f'A {describe_size(w, h)} triangle that is {describe_position(x, y)}'

class Frame(Element):
    def __init__(self, **props):
        super().__init__('Frame', **props)

class Stack(Element):
    def __init__(self, direc, **props):
        tag = 'VStack' if direc == 'v' else 'HStack'
        super().__init__(tag, **props)
        self.direc = direc

class HStack(Stack):
    def __init__(self, **props):
        super().__init__('h', **props)

class VStack(Stack):
    def __init__(self, **props):
        super().__init__('v', **props)
