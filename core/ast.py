class AST:
    # TODO: Add position data
    def __repr__(self):
        return '{}'.format(type(self).__name__)

class Nil(AST):
    pass

NIL = Nil()

class Fail(AST):
    pass

FAIL = Fail()

##class Error(AST):
##    def __init__(self, var, arity):
##        self.var = var
##        self.arity = arity
##
##    def __repr__(self):
##        return '({} {} {})'.format(type(self).__name__, self.var, self.arity)

class Error(AST):
    pass

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

class Global(Var):
    pass

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
        self.rhs = rhs

    def __repr__(self):
        return '({} {} {})'.format(type(self).__name__, self.lhs, self.rhs)

class Function(AST):
    def __init__(self, arg, exp):
        self.arg = arg  # AST or None
        self.exp = exp

    def __repr__(self):
        return '({} {} {})'.format(type(self).__name__, self.arg, self.exp)

class Lambda(Function):
    pass

class Proc(Function):
    pass

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

class Pair(AST):
    def __init__(self, head, tail):
        """Isomorphic to List."""
        self.head = head
        self.tail = tail

    def __repr__(self):
        return '({} {} {})'.format(type(self).__name__, self.head, self.tail)

class Scope(AST):
    def __init__(self, bindings, exp):
        self.bindings = bindings
        self.exp = exp

    def __repr__(self):
        return '({} {} {})'.format(type(self).__name__, self.bindings, self.exp)

class Let(Scope):
    pass

class LetRec(Scope):
    pass

class LetStar(Scope):
    pass

class Begin(AST):
    def __init__(self, exps):
        self.exps = exps

    def __repr__(self):
        return '({} {})'.format(type(self).__name__, self.exps)

class Module(Begin):
    pass

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

class UnOp(AST):
    def __init__(self, f, exp):
        self.f = f
        self.exp = exp

    def __repr__(self):
        return '({} {} {})'.format(type(self).__name__, self.f, self.exp)

class PrefixOp(UnOp):
    pass

class PostfixOp(UnOp):
    pass

class MatchfixOp(UnOp):
    pass

class Quantifier(AST):
    def __init__(self, f, x, xs, p):
        """f x ∈ xs p"""
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
        return '({})'.format(' '.join(map(str, (type(self).__name__, self.lhs, self.rhs))))
