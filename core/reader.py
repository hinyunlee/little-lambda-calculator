# Most of the magic happen here.

from . import ast
from .lexer import Lexer
from .parser import Parser
from .data import NIL, TRUE, FALSE, Complex
from .evaluator import ls, ap, REIFIED_NIL, REIFIED_TRUE, REIFIED_FALSE
from fractions import Fraction
from functools import reduce, partial
from itertools import groupby, chain
from .fp import foldr

def chainLambda(args, exp):
    """λ arg_1 arg_2 ... arg_n . exp"""
    return foldr(lambda arg, rhs: ast.Lambda(arg, rhs), exp, args)

def chainApply(*exps):
    """exp_1 exp_2 ... exp_n"""
    return reduce(lambda lhs, exp: ast.Apply(lhs, exp), exps)

def chainList(*exps):
    return foldr(lambda exp, rhs: ast.Pair(exp, rhs), ast.NIL, exps)

def listToPairs(x):
    return foldr(lambda x, a: ast.Pair(x, a), x.tail, x.exps)

def pairsToTuple(x):
    xs = []

    while isinstance(x, ast.Pair):
        xs.append(x.head)
        x = x.tail

    return xs

def extractArgs(x):
    if isinstance(x, ast.List):
        x = listToPairs(x)

    if isinstance(x, ast.Pair):
        return [*extractArgs(x.head), *extractArgs(x.tail)]
    elif isinstance(x, ast.Tuple):
        return chain(*(extractArgs(exp) for exp in x.exps))
##    elif isinstance(x, ast.Set):
##        return chain(*(extractArgs(exp) for exp in x.exps))
    elif isinstance(x, ast.Var):
        return [x]
    else:
        return []

def enumerateArgs(x, prefix='x', i=0):
    """Replace variables in a structured arg with numbered identifiers in the order they appear from left to right.
    Constants are left as-is.

    ex. ((a, b), c) -> ((x0, x1), x2)
    """
    def traverse(x, i):
        if isinstance(x, ast.List):
            x = listToPairs(x)

        if isinstance(x, ast.Pair):
            (h, i) = traverse(x.head, i)
            (t, i) = traverse(x.tail, i)
            return (ast.Pair(h, t), i)
        elif isinstance(x, ast.Tuple):
            (xs, i) = traverse(ast.List(x.exps, ast.NIL), i)
            return (ast.Tuple(pairsToTuple(xs)), i)
##        elif isinstance(x, ast.Set):
##            (xs, i) = traverse(ast.List(x.exps, ast.NIL), i)
##            return (ast.Set(pairsToTuple(xs)), i)
        elif isinstance(x, ast.Apply):
            (l, i) = traverse(x.lhs, i)
            (r, i) = traverse(x.rhs, i)
            return (ast.Apply(l, r), i)
        elif isinstance(x, ast.Var):
            return (ast.Var('{}{}'.format(prefix, i)), i + 1)
        else:
            return (x, i)

    (x, i) = traverse(x, i)
    return x

def processLambda(arg, exp, fail=ast.FAIL):
    """Destructure lambda arguments.
    f(λa1 a2 ... an.body) → f(λa1.f(λa2. ... f(λan.body)))
    f(λ[x . y].body) → λv.if atom? v then FAIL else (f(λx y.body)) (head v) (tail v)
                     → λv.if atom? v then FAIL else (λx.λy.body) (head v) (tail v)
    f(λ(a1, a2, ..., an).body) → λv.(f(λa1 a2 ... an.body)) (v 1) (v 2) ... (v n)
                         → λv.(λa1.λa2.f(λ... an.body)) (v 1) (v 2) ... (v n)
    f(λA.body) → λv.if v = A then body else FAIL, where A is a constant

    ie.
    f(λ[x].body) → λv.if atom? v then FAIL else (f(λx.f(λ[].body))) (head v) (tail v)
                 → λv.if atom? v then FAIL else (λx.λv.if v = [] then body else FAIL) (head v) (tail v)
    f(λ[x y]).body) → λv.if atom? v then FAIL else (f(λx [y].body)) (head v) (tail v)
                    → λv.if atom? v then FAIL else (λx.f(λ[y].body)) (head v) (tail v)
                    → λv.if atom? v then FAIL else (λx.λv.if atom? v then FAIL else (λy.λv.if v = [] then body else FAIL) (head v) (tail v)) (head v) (tail v)
    """
    if isinstance(arg, (tuple, list)):
        return foldr(lambda arg, rhs: processLambda(arg, rhs), exp, arg)

    if isinstance(arg, ast.List):
        arg = listToPairs(arg)

    if isinstance(arg, ast.Pair):
        hd = lambda x: ast.Apply(ast.Var('head'), x)
        tl = lambda x: ast.Apply(ast.Var('tail'), x)
        p = ast.Apply(ast.Var('atom?'), ast.Var(1))
        a = fail
        b = chainApply(processLambda([arg.head, arg.tail], exp), hd(ast.Var(1)), tl(ast.Var(1)))
        return ast.Lambda(None, ast.If(p, a, b))
    elif isinstance(arg, ast.Tuple):
        args = arg.exps
        at = lambda x, i: ast.Apply(x, ast.Number(i))
        body = chainApply(processLambda(args, exp), *(at(ast.Var(1), i + 1) for i in range(len(args))))
        return ast.Lambda(None, body)
    elif isinstance(arg, (ast.Nil, ast.Boolean, ast.Number, ast.String)):
        p = ast.BinOp(ast.Var('_=_'), ast.Var(1), arg)
        a = exp
        b = fail
        return ast.Lambda(None, ast.If(p, a, b))
    else:
        return ast.Lambda(arg, exp)

def processOperator(x):
    """x op y → op x y"""
    if isinstance(x, ast.BinOp):
        return chainApply(x.f, processOperator(x.lhs), processOperator(x.rhs))
    elif isinstance(x, (ast.PrefixUnOp, ast.PostfixUnOp, ast.MatchfixUnOp)):
        return chainApply(x.f, processOperator(x.exp))
    elif isinstance(x, ast.Grouping):
        return processOperator(x.exp)
    else:
        return x

def handlePatternMatching(bindings):
    """[ast.Def] → [ast.Def]"""
    def destructureDef(x):
        """Definition → (name, [arg], body)"""
        lhs = processOperator(x.var)
        args = []

        while isinstance(lhs, ast.Apply):
            args.append(lhs.rhs)
            lhs = lhs.lhs

        return (lhs, tuple(reversed(args)), x.exp)

    def makeClause(arity, args, exp):
        n = len(args)
        makeFail = lambda i: foldr(lambda lhs, rhs: ast.Lambda(None, rhs), ast.FAIL, range(n - i - 1))
        f = foldr(lambda i_arg, rhs: processLambda(i_arg[1], rhs, makeFail(i_arg[0] + (arity - len(args)))), exp, tuple(enumerate(args)))
        return reduce(lambda exp, i: ast.Apply(exp, ast.Var(i)), range(arity, 0, -1), f)  # Application

    def joinPatterns(arity, patterns, error):
        clauses = [makeClause(arity, args, exp) for (_, args, exp) in patterns]
        return foldr(lambda lhs, rhs: ast.Or(lhs, rhs), error, clauses)

    def processPatterns(name, patterns):
        if len(patterns) == 1:
            (_, args, exp) = patterns[0]

            if len(args) == 0:
                val = exp
            else:
                val = processLambda(args, exp)  # f x y = a → f = λx.λy.a
        else:
            arity = max(len(args) for (_, args, _) in patterns)
            error = ast.ERROR
            val = foldr(lambda arg, exp: ast.Lambda(None, exp), joinPatterns(arity, patterns, error), range(arity))

        return ast.Def(name, val)

    bindings = [destructureDef(binding) for binding in bindings]  # [(name, args, exp)]
    groups = groupby(bindings, lambda name_args_exp: name_args_exp[0])  # [(name, [(name, args, exp)])]
    return [processPatterns(name, tuple(patterns)) for (name, patterns) in groups]  # [definition]

def makeRecScope(bindings, body):
    """Recursive scope.
    let rec
        g1 := λx1.a1
        g2 := λx2.a2
        ...
        gn := λxn.an
    in
        body
    →
    Using assignment:
        let g1 := nil in let g2 := nil in let ... in let gn := nil in begin g1 ← λx1.a1; g2 ← λx2.a2; ...; gn ← λxn.an end

    Using Y combinators (disabled):
        Y* (λg1 g2 ... gn.body) (λg1 g2 ... gn x1.a1) (λg1 g2 ... gn x2.a2) ... (λg1 g2 ... gn xn.an)
        ... or when there is only one binding:
        let rec f := λx.a in body → (λf.body) (Y (λf.λx.a))
    """
##    # FIXME: Without dependency analysis, Polyvariadic Y combinator is annoyingly slow!
##    if len(bindings) == 1:
##        binding = bindings[0]
##        f = processLambda(binding.var, binding.exp)
##        return ast.Let([ast.Def(binding.var, ast.Apply(ast.Var('Y'), f))], body)
##    else:
##        names = ast.List([binding.var for binding in bindings])
##        fs = ast.List([processLambda(names, binding.exp) for binding in bindings])
##        return ast.Let([ast.Def(names, ast.Apply(ast.Var('Y*'), fs))], body)

    def chainAssign(bindings, body):
        """var_1 ← exp_1; var_2 ← exp_2; ...; var_n ← exp_n; body"""
        joinBinding = lambda b, exp: makeSeq(ast.Assign(b.var, b.exp), exp)
        return foldr(joinBinding, body, bindings)

    assigns = chainAssign(bindings, body)
    vars = tuple(chain(*(extractArgs(binding.var) for binding in bindings)))
    lambdas = processLambda(vars, assigns)
    return chainApply(lambdas, *[ast.NIL]*len(vars))

def makeSeq(lhs, rhs):
    """begin lhs; rhs end"""
    return chainApply(ast.Proc(None, rhs), lhs)

def makeBody(exps):
    """begin exp_1; exp_2; ...; exp_n end"""
    if len(exps) == 0:
        return ast.NIL
    else:
        return foldr(makeSeq, exps[-1], exps[:-1])

def makeLabel(k, v):
    """label k v"""
    return chainApply(ast.Var('label'), k, v)

def transformNamed(exp):
    """Turn named variables into De Bruijn indices."""
    def traverse(x, stack):
        if isinstance(x, ast.Apply):
            return ast.Apply(traverse(x.lhs, stack), traverse(x.rhs, stack))
        elif isinstance(x, ast.Pair):
            return ast.Pair(traverse(x.head, stack), traverse(x.tail, stack))
        elif isinstance(x, ast.Tuple):
            return ast.Tuple([traverse(exp, stack) for exp in x.exps])
        elif isinstance(x, ast.Set):
            return ast.Set([traverse(exp, stack) for exp in x.exps])
        elif isinstance(x, ast.Lambda):
            return ast.Lambda(None, traverse(x.exp, [x.arg, *stack]))
        elif isinstance(x, ast.Proc):
            return ast.Proc(None, traverse(x.exp, [x.arg, *stack]))
        elif isinstance(x, ast.Assign):
            return ast.Assign(traverse(x.lhs, stack), traverse(x.rhs, stack))
        elif isinstance(x, ast.Var):
            if x.name == 'reify':
                return x
            else:
                i = next((i for (i, y) in enumerate(stack, 1) if x == y), None)
                return ast.Var(i) if i else ast.Global(x.name) if isinstance(x.name, str) else x
        else:
            return x

    return traverse(exp, [])

def transformEnriched(exp):
    """Enriched lambda calculus → Lambda calculus"""
    def traverse(x, count, offset, onFail):
        """Counts the number of arguments in the environment since the start of a pattern match operation.
        Offsets out of bound variables."""
        if isinstance(x, ast.Apply):
            return ast.Apply(traverse(x.lhs, count, offset, onFail), traverse(x.rhs, count, offset, onFail))
        elif isinstance(x, ast.Pair):
            return ast.Pair(traverse(x.head, count, offset, onFail), traverse(x.tail, count, offset, onFail))
        elif isinstance(x, ast.Tuple):
            return ast.Tuple([traverse(exp, count, offset, onFail) for exp in x.exps])
        elif isinstance(x, ast.Set):
            return ast.Set([traverse(exp, count, offset, onFail) for exp in x.exps])
        elif isinstance(x, ast.Lambda):
            return ast.Lambda(x.arg, traverse(x.exp, count + 1, offset, onFail))
        elif isinstance(x, ast.Proc):
            return ast.Proc(x.arg, traverse(x.exp, count + 1, offset, onFail))
        elif isinstance(x, ast.Assign):
            return ast.Assign(traverse(x.lhs, count, offset, onFail), traverse(x.rhs, count, offset, onFail))
        elif isinstance(x, ast.Or):
            return ast.Apply(ast.Lambda(None, traverse(x.lhs, 0, offset + 1, ast.Var)), traverse(x.rhs, count, offset, onFail))
        elif isinstance(x, (ast.Fail, ast.Error)):
            return onFail(count + 1)
        elif isinstance(x, ast.Var):
            if isinstance(x.name, int):
                if x.name > count:
                    return ast.Var(x.name + offset)

            return x
        else:
            return x

    onFail = lambda _: ast.Apply(ast.Global('error'), ast.String('Pattern match failure'))
    return traverse(exp, 0, 0, onFail)

def transformLet(bindings, exp):
    """let x = a and y = b and ... in ... → (λx.λy. ...) a b"""
    bindings = handlePatternMatching(bindings)
    (args, exps) = zip(*((binding.var, binding.exp) for binding in bindings))
    lambdas = processLambda(args, exp)
    return transformCore(chainApply(lambdas, *exps))

def transformLetRec(bindings, exp):
    """let rec x1 = a and x2 and ... = b in ... → Y* (λ x1 x2 ... xn. ...) (λ x1 x2 ... xn. a) (λ x1 x2 ... xn. b) ..."""
    return transformCore(makeRecScope(handlePatternMatching(bindings), exp))

def transformLetStar(bindings, exp):
    """let* x = a and y = b and ... in ... → let x = a in let y = b in ..."""
    return transformCore(foldr(lambda x, a: ast.Let([x], a), exp, bindings))

def transformApply(lhs, rhs):
    if isinstance(rhs, (tuple, list)):
        x = transformCore(chainApply(lhs, *rhs))
        (lhs, rhs) = (x.lhs, x.rhs)

    return ast.Apply(transformCore(lhs), transformCore(rhs))

def transformBegin(exps):
    """begin exp_1; exp_2; ...; exp_n end → let rec ... in ..."""
    bindings = handlePatternMatching([exp for exp in exps if isinstance(exp, ast.Def)])
    body = [exp for exp in exps if not isinstance(exp, ast.Def)]
    return transformCore(makeRecScope(bindings, makeBody(body)))

def transformModule(exps):
    """Top level definitions."""
    bindings = handlePatternMatching([exp for exp in exps if isinstance(exp, ast.Def)])
    vars = tuple(chain(*(extractArgs(binding.var) for binding in bindings)))
    labels = [makeLabel(ast.String(var.name), var) for var in vars]
    body = [exp for exp in exps if not isinstance(exp, ast.Def)]
    return transformCore(makeRecScope(bindings, makeBody([*labels, *body])))

def transformLambda(arg, exp):
    if isinstance(arg, (tuple, list)):
        if len(arg) > 0:
            return transformCore(chainLambda(arg, exp))
        else:
            return transformLambda(None, exp)
    else:
        x = processLambda(arg, exp)
        return ast.Lambda(transformCore(x.arg), transformCore(x.exp))

def transformAssign(lhs, rhs):
    if isinstance(lhs, (ast.List, ast.Pair, ast.Tuple)):
        # Bind value to temp variables then assign to target variables
        argList = extractArgs(lhs)
        tempArg = enumerateArgs(lhs, 'x', 1)
        body = makeBody([ast.Assign(arg, ast.Var('x{}'.format(i))) for (i, arg) in enumerate(argList, 1)])
        return transformCore(ast.Apply(processLambda(tempArg, body), rhs))
    else:
        return ast.Assign(transformCore(lhs), transformCore(rhs))

def transformCore(x):
    """Core Language Core AST → Enriched Lambda Calculus AST"""
    if isinstance(x, ast.Grouping):
        return transformCore(x.exp)
    elif isinstance(x, ast.Apply):
        return transformApply(x.lhs, x.rhs)
    elif isinstance(x, ast.Lambda):
        return transformLambda(x.arg, x.exp)
    elif isinstance(x, ast.Proc):
        return ast.Proc(transformCore(x.arg), transformCore(x.exp))
    elif isinstance(x, ast.If):
        return chainApply(ast.Var('if'), transformCore(x.p), transformCore(x.a), transformCore(x.b))
    elif isinstance(x, ast.Let):
        return transformLet(x.bindings, x.exp)
    elif isinstance(x, ast.LetRec):
        return transformLetRec(x.bindings, x.exp)
    elif isinstance(x, ast.LetStar):
        return transformLetStar(x.bindings, x.exp)
    elif isinstance(x, ast.List):
        return foldr(ast.Pair, transformCore(x.tail), [transformCore(exp) for exp in x.exps])
    elif isinstance(x, ast.Tuple):
        return ast.Tuple([transformCore(exp) for exp in x.exps])
    elif isinstance(x, ast.Set):
        return ast.Set([transformCore(exp) for exp in x.exps])
    elif isinstance(x, ast.Begin):
        return transformBegin(x.exps)
    elif isinstance(x, ast.Module):
        return transformModule(x.exps)
    elif isinstance(x, ast.Assign):
        return transformAssign(x.lhs, x.rhs)
    elif isinstance(x, ast.BinOp):
        return transformCore(chainApply(x.f, x.lhs, x.rhs))
    elif isinstance(x, (ast.PrefixUnOp, ast.PostfixUnOp, ast.MatchfixUnOp)):
        return transformCore(chainApply(x.f, x.exp))
    elif isinstance(x, ast.Quantifier):
        return transformCore(chainApply(x.f, ast.Lambda(x.x, x.p), x.xs))
    elif isinstance(x, ast.Or):
        return ast.Or(transformCore(x.lhs), transformCore(x.rhs))
    else:
        return x

def reify(x):
    """AST → Data (Code)"""
    def reifyBoolean(x):
        if x:
            return REIFIED_TRUE
        else:
            return REIFIED_FALSE

    def reifyNumber(x):
        if isinstance(x, int):
            return ls('Integer', x)
        elif isinstance(x, float):
            return ls('Float', x)
        elif isinstance(x, Fraction):
            return ls('Rational', reifyNumber(x.numerator), reifyNumber(x.denominator))
        elif isinstance(x, Complex):
            return ls('Complex', reifyNumber(x.real), reifyNumber(x.imaginary))
        else:
            raise Exception('Failed to reify number', x)

    def reifyApply(lhs, rhs):
        if isinstance(lhs, ast.Var) and lhs.name == 'reify':
            return ls('Reify', reify(rhs))
        else:
            return ls('Apply', reify(lhs), reify(rhs))

    if isinstance(x, ast.Nil):
        return REIFIED_NIL
    elif isinstance(x, ast.Boolean):
        return reifyBoolean(x.value)
    elif isinstance(x, ast.Number):
        return reifyNumber(x.value)
    elif isinstance(x, ast.String):
        return ls('String', x.value)
    elif isinstance(x, ast.Apply):
        return reifyApply(x.lhs, x.rhs)
    elif isinstance(x, ast.Var):
        return ls('Var', x.name)
    elif isinstance(x, ast.Global):
        return ls('Global', x.name)
    elif isinstance(x, ast.Lambda):
        if x.arg is None:
            return ls('Lambda', reify(x.exp))
        else:
            return ls('Lambda', reify(x.arg), reify(x.exp))
    elif isinstance(x, ast.Proc):
        if x.arg is None:
            return ls('Proc', reify(x.exp))
        else:
            return ls('Proc', reify(x.arg), reify(x.exp))
    elif isinstance(x, ast.Pair):
        return ls('Pair', reify(x.head), reify(x.tail))
    elif isinstance(x, ast.Tuple):
        return ls('Tuple', *(reify(exp) for exp in x.exps))
    elif isinstance(x, ast.Set):
        return ls('Set', *(reify(exp) for exp in x.exps))
    elif isinstance(x, ast.Assign):
        return ls('Assign', reify(x.lhs), reify(x.rhs))
    else:
        raise Exception('Failed to reify', x)

class Reader:
    def __init__(self):
        self.lexer = Lexer()
        self.parser = Parser()

    def read(self, x):
        return reify(transformNamed(transformEnriched(transformCore(self.parser.parse(self.lexer.lex(x))))))