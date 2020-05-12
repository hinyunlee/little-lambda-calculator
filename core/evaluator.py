# KEK machine
# Krivine machine + CEK machine = KEK machine

from fractions import Fraction
from functools import reduce, partial
from .data import NIL, TRUE, FALSE, Complex, Pair, Set#, FAIL
from .fp import foldr

ls = lambda *exps: foldr(Pair, NIL, exps)
ap = lambda *exps: reduce(partial(ls, 'Apply'), exps)
lm = lambda args, exp: foldr(partial(ls, 'Lambda'), exp, args)

class Kont:
    def __init__(self, k):
        self.k = k

    def __repr__(self):
        return '({} ...)'.format(type(self).__name__)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.k == other.k

class Closure:
    def __init__(self, exp, env):
        self.exp = exp
        self.env = env

    def __repr__(self):
        if isinstance(self.exp, Function):
            return '({} ...)'.format(type(self).__name__)
        else:
            return '(Thunk ...)'

# C

class Control:
    pass

class Var(Control):
    def __init__(self, name):
        self.name = name  # String or Integer (De Bruijn index)

    def __repr__(self):
        if isinstance(self.name, str):
            return self.name
        else:
            return "*{}".format(self.name)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.name == other.name

    def reify(self):
        return ls(type(self).__name__, self.name)

class Global(Var):
    pass

class Apply(Control):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return '({} {})'.format(str(self.lhs)[1:-1] if isinstance(self.lhs, Apply) else self.lhs, self.rhs)

    def __eq__(self, other):
        return  isinstance(other, self.__class__) and self.lhs == other.lhs and self.rhs == other.rhs

    def reify(self):
        return ls(type(self).__name__, reify(self.lhs), reify(self.rhs))

class Cons(Control):
    def __init__(self, head, tail):
        self.head = head
        self.tail = tail

    def __repr__(self):
        return '[{} . {}]'.format(self.head, self.tail)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.head == other.head and self.tail == other.tail

    def reify(self):
        return ls('Pair', reify(self.head), reify(self.tail))

class MakeTuple(Control):
    def __init__(self, exps):
        self.exps = exps

    def __repr__(self):
        return '({})'.format(', '.join(self.exps))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and len(self.exps) == len(other.exps) and all(x == y for (x, y) in zip(self.exps, other.exps))

    def reify(self):
        return ls('Tuple', *(reify(exp) for exp in self.exps))

class MakeSet(Control):
    def __init__(self, exps):
        self.exps = exps

    def __repr__(self):
        return '({})'.format(', '.join(self.exps))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and len(self.exps) == len(other.exps) and all(x == y for (x, y) in zip(self.exps, other.exps))

    def reify(self):
        return ls('Set', *(reify(exp) for exp in self.exps))

class Function(Control):
    def __init__(self, exp):
        self.exp = exp

    def __eq__(self, other):
        return isinstance(other, self.__class__)  and self.exp == other.exp

    def reify(self):
        if isinstance(self.exp, Mapping):
            return ls(type(self).__name__, reify(self.exp.var), reify(self.exp.exp))
        else:
            return ls(type(self).__name__, reify(self.exp))

class Lambda(Function):
    def __repr__(self):
        exp = str(self.exp)[1:-1] if isinstance(self.exp, (Lambda, Apply)) else self.exp
        return '(λ{})'.format(exp)

class Proc(Function):
    """Call-by-value lambda."""
    def __repr__(self):
        exp = str(self.exp)[1:-1] if isinstance(self.exp, (Lambda, Apply)) else self.exp
        return '(to {})'.format(exp)

class Reify(Control):
    def __init__(self, exp):
        self.exp = exp

    def __repr__(self):
        return '({} {})'.format(type(self).__name__, self.exp)

    def reify(self):
        return ls(type(self).__name__, reify(self.exp))

class Assign(Control):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return '({} ← {})'.format(self.lhs, self.rhs)

    def reify(self):
        return ls(type(self).__name__, reify(self.lhs), reify(self.rhs))

# E

class Mapping:
    def __init__(self, var, exp):
        self.var = var
        self.exp = exp

    def __repr__(self):
        return '{}.{}'.format(self.var, self.exp)       

class Env:
    def __init__(self, exp, parent):
        self.exp = exp
        self.parent = parent

    def __repr__(self):
        return '{}::{}'.format(self.exp, self.parent or '()')

# K

class Continuation:
    pass

class Arg(Continuation):
    """_ x"""
    def __init__(self, c, e, k):
        self.c = c
        self.e = e
        self.k = k

    def __repr__(self):
        return '({} {})::{}'.format(type(self).__name__, self.c, self.k or '()')

class Call(Continuation):
    """f _"""
    def __init__(self, c, e, k):
        self.c = c  # Function
        self.e = e
        self.k = k

    def __repr__(self):
        return '({} {})::{}'.format(type(self).__name__, self.c, self.k or '()')

class Memoize(Continuation):
    def __init__(self, f, k):
        self.f = f
        self.k = k

    def __repr__(self):
        return '{}::{}'.format(type(self).__name__, self.k or '()')

class Slot(Continuation):
    """_ _ ← x"""
    def __init__(self, c, e, k):
        self.c = c
        self.e = e
        self.k = k

    def __repr__(self):
        return '({} {})::{}'.format(type(self).__name__, self.c, self.k or '()')

# Code ↔ Data

REIFIED_NIL = ls('Nil')
REIFIED_TRUE = ls('Boolean', 'True')
REIFIED_FALSE = ls('Boolean', 'False')

def pairsToList(x):
    xs = []

    while isinstance(x, Pair):
        xs.append(x.head)
        x = x.tail

    return xs

def reify(x):
    if x is NIL:
        return REIFIED_NIL
    elif x is FALSE:
        return REIFIED_TRUE
    elif x is TRUE:
        return REIFIED_FALSE
    elif isinstance(x, int):
        return ls('Integer', x)
    elif isinstance(x, float):
        return ls('Float', x)
    elif isinstance(x, Fraction):
        return ls('Rational', reify(x.numerator), reify(x.denominator))
    elif isinstance(x, Complex):
        return ls('Complex', reify(x.real), reify(x.imaginary))
    elif isinstance(x, str):
        return ls('String', x)
    elif isinstance(x, Control):
        return x.reify()
    else:
        raise Exception('Failed to reify', x)

def compile(x):
    h = x.head

    if h == 'Nil':  # [Nil]
        return NIL
    elif h == 'Boolean':
        h2 = x.tail.head

        if h2 == 'True':  # [Boolean True]
            return TRUE
        elif h2 == 'False':  # [Boolean False]
            return FALSE
        else:
            raise Exception('Invalid boolean value', h2)
    elif h == 'Integer':  # [Integer x]
        return x.tail.head
    elif h == 'Float':  # [Float x]
        return x.tail.head
    elif h == 'Rational':  # [Rational numerator denominator]
        x2 = x.tail
        x3 = x2.tail
        return Fraction(compile(x2.head), compile(x3.head))
    elif h == 'Complex':  # [Complex real imaginary]
        x2 = x.tail
        x3 = x2.tail
        return Complex(compile(x2.head), compile(x3.head))
    elif h == 'String':  # [String x]
        return x.tail.head
    elif h == 'Pair':  # [Pair head tail]
        x2 = x.tail
        x3 = x2.tail
        return Cons(compile(x2.head), compile(x3.head))
    elif h == 'Tuple':  # [Tuple x1 x2 ... xn]
        return MakeTuple([compile(x) for x in pairsToList(x.tail)])
    elif h == 'Set':  # [Set x1 x2 ... xn]
        return MakeSet([compile(x) for x in pairsToList(x.tail)])
    elif h == 'Var':  # [Var x]
        return Var(x.tail.head)
    elif h == 'Global':  # [Global x]
        return Global(x.tail.head)
    elif h == 'Apply':  # [Apply lhs rhs]
        x2 = x.tail
        x3 = x2.tail
        return Apply(compile(x2.head), compile(x3.head))
    elif h == 'Lambda':  # [Lambda arg? exp]
        x2 = x.tail
        x3 = x2.tail

        if x3 is NIL:
            return Lambda(compile(x2.head))
        else:
            return Lambda(Mapping(compile(x2.head), compile(x3.head)))
    elif h == 'Proc':  # [Proc arg? exp]
        x2 = x.tail
        x3 = x2.tail

        if x3 is NIL:
            return Proc(compile(x2.head))
        else:
            return Proc(Mapping(compile(x2.head), compile(x3.head)))
    elif h == 'Reify':  # [Reify x]
        return Reify(compile(x.tail.head))
    elif h == 'Assign':  # [Assign lhs rhs]
        x2 = x.tail
        x3 = x2.tail
        return Assign(compile(x2.head), compile(x3.head))
    else:
        raise Exception('Failed to compile', x)

# Environment

def clos(exp, env):
    if isRedex(exp) or isinstance(exp, Function):
        return Closure(exp, env)
    else:
        return exp

def setEnv(env, exp):
    env.exp = exp

def setMapping(env, exp):
    env.exp.exp = exp

def setHead(x, exp):
    x.head = exp

def setTail(x, exp):
    x.tail = exp

def setSlot(o, k, exp):
    o[k] = exp

def findEnvByName(name, env):
    while env:
        exp = env.exp

        if isinstance(exp, Mapping) and exp.var.name == name:
            return env

        env = env.parent

def findEnvByIndex(i, env):
    while i > 1:
        i = i - 1
        env = env.parent

    return env

# Reduction

def betaReduce(lhs, lhsEnv, rhs, rhsEnv):
    """β-reduction: (λV.M)N → M[V := N]"""
    exp = lhs.exp

    if isinstance(exp, Mapping):
        return (exp.exp, Env(Mapping(exp.var, clos(rhs, rhsEnv)), lhsEnv))
    else:
        return (exp, Env(clos(rhs, rhsEnv), lhsEnv))

def isRedex(x):
    return isinstance(x, (Var, Apply, Cons, MakeTuple, MakeSet, Assign))

class Evaluator:
    def __init__(self, builtins=None, prelude=None):
        """Lambda calculus machine.
        builtins -- Dictionary of built-in values.
        prelude -- Prelude program to be evaluated.
        """
        self.root = builtins
        self.eval(prelude)
        self.prelude = self.root

    def reset(self):
        self.root = self.prelude

    def eval(self, exp):
        return self.exec(compile(exp))

    def apply(self, c, e, k):
        if isinstance(k.c, Proc):
            (c, e) = betaReduce(k.c, k.e, c, e)
            k = k.k
        else:
            if isinstance(k.c, Pair):  # Head/tail of pair
                if c == 'head' or c == 1:
                    c = k.c.head
                    memo = partial(setHead, k.c)
                elif c == 'tail' or c == 2:
                    c = k.c.tail
                    memo = partial(setTail, k.c)
                else:
                    raise Exception('Cannot access pair', type(c).__name__, c, type(k).__name__, k)
            elif isinstance(k.c, (tuple, list)):
                if isinstance(c, int):
                    i = c - 1
                    c = k.c[i]
                    memo = partial(setSlot, k.c, i)
                else:
                    raise Exception('Cannot access tuple', type(c).__name__, c, type(k).__name__, k)
            else:
                raise Exception('Cannot call', type(c).__name__, c, type(k).__name__, k)

            k = k.k

            if isinstance(k, Slot):  # a(x) ← b
                memo(clos(k.c, e))
                (c, k) = (NIL, k.k)
            else:
                if isinstance(c, Closure):
                    (c, e) = (c.exp, c.env)
                
                if isRedex(c):
                    k = Memoize(memo, k)

        return (c, e, k)

    def lookup(self, c, e):
        key = c.name

        if isinstance(c, Global):
            result = self.root.get(key, key)  # Symbol if undefined 
            memo = partial(setSlot, self.root, key)
        else:
            if isinstance(key, str):
                env = findEnvByName(key, e)
            elif isinstance(key, int):  # [1, n]
                env = findEnvByIndex(key, e)
            else:
                raise Exception('Invalid binding', key)

            if env:
                exp = env.exp

                if isinstance(exp, Mapping):
                    result = exp.exp
                    memo = partial(setMapping, env)
                else:
                    result = exp
                    memo = partial(setEnv, env)
            else:
                result = self.root.get(key, key)
                memo = partial(setSlot, self.root, key)

        return (result, memo)

    def exec(self, x):
        (c, e, k) = (x, None, None)

        while True:
            if callable(c):
                (c, e, k) = c(c, e, k, self)

                if isinstance(c, Closure):
                    (c, e) = (c.exp, c.env)
            elif isinstance(c, Var):
                (c, memo) = self.lookup(c, e)

                if isinstance(c, Closure):
                    (c, e) = (c.exp, c.env)

                if isRedex(c):
                    k = Memoize(memo, k)
            elif isinstance(c, Apply):
                (c, k) = (c.lhs, Arg(c.rhs, e, k))
            elif isinstance(c, Cons):
                c = Pair(clos(c.head, e), clos(c.tail, e))
            elif isinstance(c, MakeTuple):
                c = [clos(exp, e) for exp in c.exps]
            elif isinstance(c, MakeSet):
                c = Set([clos(exp, e) for exp in c.exps])
            elif isinstance(c, Reify):
                c = reify(c.exp)
            elif isinstance(c, Assign):
                dest = c.lhs

                if isinstance(dest, Var):
                    (_, memo) = self.lookup(dest, e)
                    memo(clos(c.rhs, e))
                    c = NIL
                elif isinstance(dest, Apply):
                    (c, k) = (dest, Slot(c.rhs, e, k))
                else:
                    raise Exception('Cannot assign', type(c).__name__, c)
            elif isinstance(k, Arg):  # Application
                if isinstance(c, Lambda):  # Call by need
                    (c, e) = betaReduce(c, e, k.c, k.e)
                    k = k.k
                elif isinstance(c, (Proc, Pair, tuple, list)):  # Call by value
                    (c, e, k) = (k.c, k.e, Call(c, e, k.k))  # Evaluate argument
                elif isinstance(c, Kont):
                    (c, e, k) = (k.c, k.e, c.k)
                else:
                    raise Exception('Cannot apply', type(c).__name__, c, type(k).__name__, k)
            elif isinstance(k, Call):  # Call by value
                (c, e, k) = self.apply(c, e, k)
            elif isinstance(k, Memoize):
                k.f(clos(c, e))
                k = k.k
            elif k is None:
                return c  # Normal form
            else:
                raise Exception('Failed to evaluate', type(c).__name__, c, type(k).__name__, k)
