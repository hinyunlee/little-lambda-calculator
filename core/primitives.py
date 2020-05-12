# Primitive operations

from . import printer
from . import reader
from .data import TRUE, FALSE, NIL, Complex, Pair, Set
from .evaluator import Env, Lambda, Proc, Var, Global, Apply, Closure, Memoize, Arg, Kont, clos, isRedex
from functools import reduce, partial
from fractions import Fraction
import random
import math
from .fp import foldr

### Y := λf.(λx.f(x x))(λx.f(x x))
##Y = Lambda(Var('f'), Apply(Lambda(Var('x'), Apply(Var('f'), Apply(Var('x'), Var('x')))),
##                           Lambda(Var('x'), Apply(Var('f'), Apply(Var('x'), Var('x'))))))
##
### map f xs := if atom? xs then nil else f (head xs) :: map f (tail xs)
##MAP = Apply(Var('Y'), lm([Var('map'), Var('f'), Var('xs')],
##                         ap(Var('if'), Apply(Var('atom?'), Var('xs')),
##                                       NIL,
##                                       ap(Var('_::_'),
##                                          Apply(Var('f'), Apply(Var('head'), Var('xs'))),
##                                          ap(Var('map'), Var('f'), Apply(Var('tail'), Var('xs')))))))
##
### Y* fs := Y (λf.map (λg. g f) fs)
##Y_POLY = Lambda(Var('fs'), Apply(Var('Y'), Lambda(Var('f'), ap(Var('map'), Lambda(Var('g'), Apply(Var('g'), Var('f'))), Var('fs')))))

r = reader.Reader()
p = printer.Printer()

ls = lambda *exps: foldr(Pair, NIL, exps)
ap = lambda *exps: reduce(Apply, exps)
lm = lambda args, exp: foldr(Lambda, exp, args)

def callcc(c, e, k, machine):
    return (e.exp, e, Arg(Kont(k), e, k))

def cond(x):
    if x is None or x is False:
        return Lambda(Lambda(Var(1)))
    else:
        return Lambda(Lambda(Var(2)))

def setHead(x, exp):
    x.head = exp

def head(c, e, k, machine):
    x = e.exp
    c = x.head

    if isinstance(c, Closure):
        (c, e) = (c.exp, c.env)

    if isRedex(c):
        k = Memoize(partial(setHead, x), k)

    return (c, e, k)

def setTail(x, exp):
    x.tail = exp

def tail(c, e, k, machine):
    x = e.exp
    c = x.tail

    if isinstance(c, Closure):
        (c, e) = (c.exp, c.env)

    if isRedex(c):
        k = Memoize(partial(setTail, x), k)

    return (c, e, k)

def isInt(x):
    if isinstance(x, int):
        return True
    elif isinstance(x, float):
        return x.is_integer()
    elif isinstance(x, Fraction):
        return x.denominator == 1
    elif isinstance(x, Complex):
        return isInt(x.real) and x.imaginary == 0
    else:
        return False

def isReal(x):
    return isinstance(x, (int, float, Fraction))

def explode(x):
    return reduce(lambda xs, x: Pair(x, xs), reversed(x), NIL)

def slurp(x):
    with open(x, 'r', encoding='utf-8') as f:
        return f.read()

def spit(x, y):
    with open(x, 'w', encoding='utf-8') as f:
        return f.write(y)

def loadFile(c, e, k, machine):
    with open(e.exp, encoding='utf-8') as f:
        c = machine.eval(r.read(f.read()))

    return (c, e, k)

def reset(c, e, k, machine):
    machine.reset()
    return (NIL, e, k)

def div(x, y):
    if isinstance(x, Complex):
        return x/y
    elif isinstance(y, Complex):
        if isinstance(x, (int, float, Fraction)):
            return Complex(x, 0)/y
        else:
            return x/y
    else:
        return Fraction(x, y)

def eval(c, e, k, machine):
    return (machine.eval(e.exp), e, k)

def exit(c, e, k, machine):
    raise SystemExit

def label(c, e, k, machine):
    e2 = e.parent
    key = e2.exp
    value = e.exp
    machine.root[key] = value
    return (NIL, e, k)

def error(x):
    raise Exception(x)

def bind(n, f):
    """Bind n-argument function."""
    def take(n, e):
        xs = []

        for _ in range(n):
            xs.append(e.exp)
            e = e.parent

        return xs

    def g(c, e, k, machine):
        args = take(n, e)
        return (f(*reversed(args)), e, k)

    body = reduce(lambda a, x: Proc(a), range(n), g)
    return reduce(lambda a, x: Lambda(Apply(a, Var(1))), range(n - 1), body)  # Final arg not delayed

def toTuple(c, e, k, machine):
    x = e.exp

    if isinstance(x, Pair):
        xs = []

        while isinstance(x, Pair):
            xs.append(x.head)
            x = x.tail

        c = xs
    elif isinstance(x, Set):
        c = x.xs
    elif isinstance(x, (tuple, list)):
        c = x
    else:
        c = NIL

    return (c, e, k)

def extend(env=None):
    xss = [
##        ('nil', NIL),
##        ('true', TRUE),
##        ('false', FALSE),
        ('_+_', bind(2, lambda x, y: x + y)),
        ('_-_', bind(2, lambda x, y: x - y)),
        ('_⋅_', bind(2, lambda x, y: x * y)),
        ('_/_', bind(2, div)),
        ('-_', bind(1, lambda x: -x)),
        ('_mod_', bind(2, lambda x, y: x % y)),
        ('_↑_', bind(2, pow)),
        ('_≤_', bind(2, lambda x, y: x <= y)),
        ('_<_', bind(2, lambda x, y: x < y)),
        ('_≥_', bind(2, lambda x, y: x >= y)),
        ('_>_', bind(2, lambda x, y: x > y)),
        ('_=_', bind(2, lambda x, y: x == y)),
        ('_≠_', bind(2, lambda x, y: x != y)),
        ('abs', bind(1, abs)),
        ('call/cc', Lambda(callcc)),
        ('strcat', bind(2, lambda x, y: ''.join((x, y)))),
        ('format', bind(1, lambda x: p.format(x))),
        ('puts', bind(1, lambda x: print(p.format(x), end='') or NIL)),
        ('head', Proc(head)),
        ('tail', Proc(tail)),
        ('if', bind(1, cond)),
        ('atom?', bind(1, lambda x: not isinstance(x, Pair))),
        ('pair?', bind(1, lambda x: isinstance(x, Pair))),
        ('list', bind(1, lambda x: ls(*x))),
        ('tuple', Proc(Apply(Proc(toTuple), Apply(Global('force-one-level'), Var(1))))),
        ('int?', bind(1, isInt)),
        ('rational?', bind(1, isReal)),
        ('real?', bind(1, isReal)),
        ('number?', bind(1, isReal)),
        ('exact?', bind(1, lambda x: isinstance(x, (int, Fraction)))),
        ('inexact?', bind(1, lambda x: isinstance(x, float))),
        ('bool', bind(1, lambda x: FALSE if x in (NIL, FALSE) else TRUE)),
        ('bool?', bind(1, lambda x: isinstance(x, bool))),
        ('str', bind(1, str)),
        ('str?', bind(1, lambda x: isinstance(x, str))),
        ('char?', bind(1, lambda x: isinstance(x, str) and len(x) == 1)),
        ('tuple?', bind(1, lambda x: isinstance(x, (tuple, list)))),
        ('set?', bind(1, lambda x: isinstance(x, Set))),
        ('eval', Proc(eval)),
        ('read', bind(1, lambda x: explode(r.read(x)))),
        ('int', bind(1, lambda x: x.numerator)),  # TODO
        ('frac', bind(1, lambda x: x.denominator)),  # TODO
        ('chr', bind(1, chr)),
        ('ord', bind(1, ord)),
        ('explode', bind(1, explode)),
        ('slurp', bind(1, slurp)),
        ('spit', bind(1, spit)),
        ('open', bind(1, lambda x: open(x, 'r+'))),
        ('close', bind(1, lambda x: x.close())),
        ('read-byte', bind(1, lambda x: x.read(1))),
        ('write-byte', bind(1, lambda x: x.write(x))),
        ('load', Proc(loadFile)),
        ('reset', Lambda(reset)),
        ('rand', bind(1, lambda x: random.random())),
        ('exit', Lambda(exit)),
        ('label', Proc(Lambda(label))),
        ('error', bind(1, error)),

##        ('Y', Y),
##        ('map', MAP),
##        ('Y*', Y_POLY),

        # Math
        ('π', math.pi),
        ('e', math.e),
        ('∞', math.inf),
        ('sin', bind(1, math.sin)),
        ('cos', bind(1, math.cos)),
        ('tan', bind(1, math.tan)),
        ('arcsin', bind(1, math.asin)),
        ('arccos', bind(1, math.acos)),
        ('arctan', bind(1, math.atan)),
        ('atan2', bind(2, math.atan2)),
        ('sinh', bind(1, math.sinh)),
        ('cosh', bind(1, math.cosh)),
        ('tanh', bind(1, math.tanh)),
        ('arsinh', bind(1, math.asinh)),
        ('arcosh', bind(1, math.acosh)),
        ('artanh', bind(1, math.atanh)),
        ('floor', bind(1, math.floor)),
        ('ceil', bind(1, math.ceil)),
        ('degrees', bind(1, math.degrees)),
        ('radians', bind(1, math.radians)),
        ('exp', bind(1, math.exp)),
        ('log', bind(2, lambda a, x: math.log(x, a))),
        ('ln', bind(1, math.log)),
        ('lg', bind(1, math.log10)),
        ('lb', bind(1, math.log2)),
        ('sqrt', bind(1, math.sqrt))
    ]

    return dict(((k, clos(v, env)) for k, v in xss))