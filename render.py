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
    return f"\"{x}\"" if type(x) is str else x

def kv(k, v):
    return k if v is True else f'{k}={{{st(v)}}}'

def kvs(props):
    return ' '.join([kv(k, v) for k, v in props.items()])

def ind(s, n=INDENT):
    pad = ' ' * n
    return '\n'.join([f'{pad}{line}' for line in s.split('\n')])

##
## describers
##

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
## classes
##

class Element:
    def __init__(self, tag='Element', unary=False, **props):
        self.tag = tag
        self.unary = unary
        self.props = props

    def inner_jsx(self, indent=0):
        return ''

    def render_jsx(self, indent=0):
        props = kvs(self.props)
        pad = ' ' if len(props) > 0 else ''
        if self.unary:
            return f'<{self.tag} {props}{pad}/>'
        else:
            inner = self.inner_jsx(indent + INDENT)
            return f'<{self.tag} {props}>{ind(inner)}</{self.tag}>'

class Group(Element):
    def __init__(self, *children, **props):
        super().__init__('Group', unary=False, **props)
        self.children = children

    def inner_jsx(self, indent=0):
        cstr = '\n'.join([c.render_jsx(indent) for c in self.children])
        return f'\n{cstr}\n' if len(cstr) > 0 else ''

class Rect(Element):
    def __init__(self, **props):
        super().__init__('Rect', **props)

class Ellipse(Element):
    def __init__(self, **props):
        super().__init__('Ellipse', **props)

class Circle(Element):
    def __init__(self, **props):
        super().__init__('Circle', **props)

class Square(Element):
    def __init__(self, **props):
        super().__init__('Square', **props)

class Triangle(Element):
    def __init__(self, **props):
        super().__init__('Triangle', **props)

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
