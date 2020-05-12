class AST:
    # TODO: Add position data
    pass

class Nil(AST):
    def __repr__(self):
        return '{}'.format(type(self).__name__)

NIL = Nil()

class Fail(AST):
    def __repr__(self):
        return '{}'.format(type(self).__name__)

FAIL = Fail()

class Error(AST):
    def __repr__(self):
        return '{}'.format(type(self).__name__)

ERROR = Error()

class Boolean(AST):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return '({} {})'.format(type(self).__name__, self.value)

TRUE = Boolean(True)
FALSE = Boolean(False)

class Number(AST):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return '({} {})'.format(type(self).__name__, self.value)

class String(AST):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return '({} {})'.format(type(self).__name__, self.value)

class Var(AST):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return '({} {})'.format(type(self).__name__, self.name)

    def __eq__(self, other):
        return isinstance(other, Var) and self.name == other.name

class Global(AST):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return '({} {})'.format(type(self).__name__, self.name)

    def __eq__(self, other):
        return isinstance(other, Var) and self.name == other.name

class Grouping(AST):
    def __init__(self, exp):
        """Grouped expression.
        Could be used for automatic code reformatting some day.
        """
        self.exp = exp

    def __repr__(self):
        return '({} {})'.format(type(self).__name__, self.exp)

class Def(AST):
    def __init__(self, var, exp):
        self.var = var
        self.exp = exp

    def __repr__(self):
        return '({} {} {})'.format(type(self).__name__, self.var, self.exp)
    
class Apply(AST):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs  # AST or [AST]

    def __repr__(self):
        return '({} {} {})'.format(type(self).__name__, self.lhs, self.rhs)

class Function(AST):
    def __init__(self, arg, exp):
        self.arg = arg  # AST or [AST] or None
        self.exp = exp

class Lambda(Function):
    def __repr__(self):
        return '({} {} {})'.format(type(self).__name__, self.arg, self.exp)

class Proc(Function):
    def __repr__(self):
        return '({} {} {})'.format(type(self).__name__, self.arg, self.exp)

class If(AST):
    def __init__(self, p, a, b):
        self.p = p
        self.a = a
        self.b = b

    def __repr__(self):
        return '({} {} {} {})'.format(type(self).__name__, self.p, self.a, self.b)

class List(AST):
    def __init__(self, exps=[], tail=NIL):
        self.exps = exps
        self.tail = tail

    def __repr__(self):
        return '({} {} {})'.format(type(self).__name__, self.exps, self.tail)

class Tuple(AST):
    def __init__(self, exps=[]):
        self.exps = exps

    def __repr__(self):
        return '({} {})'.format(type(self).__name__, self.exps)

class Set(AST):
    def __init__(self, exps=[]):
        self.exps = exps

    def __repr__(self):
        return '({} {})'.format(type(self).__name__, self.exps)

class Scope(AST):
    def __init__(self, bindings, exp):
        self.bindings = bindings
        self.exp = exp

class Pair(AST):
    def __init__(self, head, tail):
        """Isomorphic to List."""
        self.head = head
        self.tail = tail

    def __repr__(self):
        return '({} {} {})'.format(type(self).__name__, self.head, self.tail)

class Let(Scope):
    def __repr__(self):
        return '({} {} {})'.format(type(self).__name__, self.bindings, self.exp)

class LetRec(Scope):
    def __repr__(self):
        return '({} {} {})'.format(type(self).__name__, self.bindings, self.exp)

class LetStar(Scope):
    def __repr__(self):
        return '({} {} {})'.format(type(self).__name__, self.bindings, self.exp)

class Begin(AST):
    def __init__(self, exps):
        self.exps = exps

    def __repr__(self):
        return '({} {})'.format(type(self).__name__, self.exps)

class Module(AST):
    def __init__(self, exps):
        self.exps = exps

    def __repr__(self):
        return '({} {})'.format(type(self).__name__, self.exps)

class Assign(AST):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return '({} {} {})'.format(type(self).__name__, self.lhs, self.rhs)

class BinOp(AST):
    def __init__(self, f, lhs, rhs):
        self.f = f
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return '({} {} {} {})'.format(type(self).__name__, self.f, self.lhs, self.rhs)

class PrefixUnOp(AST):
    def __init__(self, f, exp):
        self.f = f
        self.exp = exp

    def __repr__(self):
        return '({} {} {})'.format(type(self).__name__, self.f, self.exp)

class PostfixUnOp(AST):
    def __init__(self, f, exp):
        self.f = f
        self.exp = exp

    def __repr__(self):
        return '({} {} {})'.format(type(self).__name__, self.f, self.exp)

class MatchfixUnOp(AST):
    def __init__(self, f, exp):
        self.f = f
        self.exp = exp

    def __repr__(self):
        return '({} {})'.format(type(self).__name__, self.exp)

class Quantifier(AST):
    def __init__(self, f, x, xs, p):
        """f x ∈ xs p(x)"""
        self.f = f
        self.x = x
        self.xs = xs
        self.p = p

    def __repr__(self):
        return '({} {} {} {} {})'.format(type(self).__name__, self.f, self.x, self.xs, self.p)

class Or(AST):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return '({} {} {})'.format(type(self).__name__, self.lhs, self.rhs)
