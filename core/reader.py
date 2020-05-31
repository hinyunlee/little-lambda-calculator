# Most of the magic happen here.

from . import ast
from .lexer import Lexer
from .parser import Parser
from .data import NIL, TRUE, FALSE, Complex
from .evaluator import ls, ap, REIFIED_NIL, REIFIED_TRUE, REIFIED_FALSE
from fractions import Fraction
from functools import reduce, partial
from itertools import groupby, chain
from .fp import foldr, foldr1

def chainLambda(args, exp):
    """λ arg_1 arg_2 ... arg_n . exp"""
    return foldr(ast.Lambda, exp, args)

def chainApply(*exps):
    """exp_1 exp_2 ... exp_n"""
    return reduce(ast.Apply, exps)

def chainList(*exps):
    return foldr(ast.Pair, ast.NIL, exps)

def listToPairs(x):
    return foldr(ast.Pair, x.tail, x.exps)

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
    f(λ[x . y].body) → λv.if pair? v then (f(λx y.body)) (head v) (tail v) else FAIL
                     → λv.if pair? v then (λx.λy.body) (head v) (tail v) else FAIL
    f(λ(a1, a2, ..., an).body) → λv.if tuple? v then (f(λa1 a2 ... an.body)) (v 1) (v 2) ... (v n) else FAIL
                               → λv.if tuple? v then (λa1.λa2.f(λ... an.body)) (v 1) (v 2) ... (v n) else FAIL
    f(λA.body) → λv.if v = A then body else FAIL, where A is a constant

    ie.
    f(λ[x].body) → λv.if pair? v then (f(λx.f(λ[].body))) (head v) (tail v) else FAIL
                 → λv.if pair? v then (λx.λv.if v = [] then body else FAIL) (head v) (tail v) else FAIL
    f(λ[x y]).body) → λv.if pair? v then (f(λx [y].body)) (head v) (tail v) else FAIL
                    → λv.if pair? v then (λx.f(λ[y].body)) (head v) (tail v) else FAIL
                    → λv.if pair? v then (λx.λv.if pair? v then (λy.λv.if v = [] then body else FAIL) (head v) (tail v) else FAIL) (head v) (tail v) else FAIL
    """
    if isinstance(arg, ast.List):
        arg = listToPairs(arg)

    if isinstance(arg, ast.Pair):
        args = (arg.head, arg.tail)
        hd = lambda x: ast.Apply(x, ast.Number(1))
        tl = lambda x: ast.Apply(x, ast.Number(2))
        p = ast.Apply(ast.Var('pair?'), ast.Var(1))
        a = chainApply(foldr(processLambda, exp, args), hd(ast.Var(1)), tl(ast.Var(1)))
        b = fail
        return ast.Lambda(None, ast.If(p, a, b))
    elif isinstance(arg, ast.Tuple):
        args = arg.exps
        at = lambda i, x: ast.Apply(x, ast.Number(i))
        eq = lambda x, y: chainApply(ast.Var('_=_'), x, y)
        an = lambda x, y: chainApply(ast.Var('_∧_'), x, y)
        card = lambda x: ast.Apply(ast.Var('card'), x)
        p = an(ast.Apply(ast.Var('tuple?'), ast.Var(1)), eq(card(ast.Var(1)), ast.Number(len(args))))
        a = chainApply(foldr(processLambda, exp, args), *(at(i + 1, ast.Var(1)) for i in range(len(args))))
        b = fail
        return ast.Lambda(None, ast.If(p, a, b))
    elif isinstance(arg, (ast.Nil, ast.Boolean, ast.Number, ast.String)):
        p = ast.BinOp(ast.Var('_=_'), ast.Var(1), arg)
        a = exp
        b = fail
        return ast.Lambda(None, ast.If(p, a, b))
    else:
        return ast.Lambda(arg, exp)

def handlePatternMatching(bindings):
    """[ast.Def] → [ast.Def]"""
    def destructureDef(x):
        """Definition → (var, [arg], body)"""
        lhs = transformCore(x.var)
        args = []

        while isinstance(lhs, ast.Apply):
            args.append(lhs.rhs)
            lhs = lhs.lhs

        return (lhs, tuple(reversed(args)), x.exp)

    def makeClause(arity, args, body):
        """(λa1 a2 ... an.body) b1 b2 ... bn"""
        makeFail = lambda i: foldr(lambda _, rhs: ast.Lambda(None, rhs), ast.FAIL, range(arity - i))
        makeLambda = lambda i, arg, rhs: processLambda(arg, rhs, makeFail(i))
        body = foldr(lambda i_arg, rhs: makeLambda(*i_arg, rhs), body, enumerate(args, 1))
        return reduce(lambda exp, i: ast.Apply(exp, ast.Var(i)), range(arity, 0, -1), body)

    def joinPatterns(patterns):
        """λb1 b2 ... bn.clause_1 | clause_2 | clause_n"""
        arity = max(len(args) for (_, args, _) in patterns)
        clauses = [makeClause(arity, args, exp) for (_, args, exp) in patterns]
        body = foldr(ast.Or, ast.ERROR, clauses)
        return foldr(lambda arg, exp: ast.Lambda(None, exp), body, range(arity))

    def processPatterns(patterns):
        if len(patterns) == 1:
            (_, args, exp) = patterns[0]
            return foldr(processLambda, exp, args)
        else:
            return joinPatterns(patterns)

    bindings = [destructureDef(binding) for binding in bindings]  # [(var, args, body)]
    groups = groupby(bindings, lambda var_args_body: var_args_body[0])  # [(var, [(var, args, body)])]
    return [ast.Def(var, processPatterns(tuple(patterns))) for (var, patterns) in groups]  # [definition]

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
    lambdas = foldr(processLambda, assigns, vars)
    return chainApply(lambdas, *[ast.NIL]*len(vars))

def makeSeq(lhs, rhs):
    """begin lhs; rhs end"""
    return chainApply(ast.Proc(None, rhs), lhs)

def makeBody(exps):
    """begin exp_1; exp_2; ...; exp_n end"""
    if len(exps) == 0:
        return ast.NIL
    else:
        return foldr1(makeSeq, exps)

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
    def find(p, x):
        """Counts the number of arguments in the environment since the start of a pattern match operation.
        Offsets out of bound variables."""
        nextFound = lambda xs: next((found for found in (find(p, x) for x in xs) if found), None)

        if p(x):
            return x
        if isinstance(x, (ast.Apply, ast.Assign, ast.Or)):
            return nextFound((x.lhs, x.rhs))
        elif isinstance(x, ast.Pair):
            return nextFound((x.head, x.tail))
        elif isinstance(x, (ast.Tuple, ast.Set)):
            return nextFound(x.exps)
        elif isinstance(x, (ast.Lambda, ast.Proc)):
            return find(p, x.exp)
        else:
            return None

##    def findIndex(p, stack):
##        return next((i for (i, y) in enumerate(stack, 1) if p(y)), None)

    def traverse(x, count, onFail):
        """Counts the number of arguments in the environment since the start of a pattern match operation.
        Offsets out of bound variables."""
        if isinstance(x, ast.Apply):
            return ast.Apply(traverse(x.lhs, count, onFail), traverse(x.rhs, count, onFail))
        elif isinstance(x, ast.Pair):
            return ast.Pair(traverse(x.head, count, onFail), traverse(x.tail, count, onFail))
        elif isinstance(x, ast.Tuple):
            return ast.Tuple([traverse(exp, count, onFail) for exp in x.exps])
        elif isinstance(x, ast.Set):
            return ast.Set([traverse(exp, count, onFail) for exp in x.exps])
        elif isinstance(x, ast.Lambda):
            return ast.Lambda(x.arg, traverse(x.exp, count + 1, onFail))
        elif isinstance(x, ast.Proc):
            return ast.Proc(x.arg, traverse(x.exp, count + 1, onFail))
        elif isinstance(x, ast.Assign):
            return ast.Assign(traverse(x.lhs, count, onFail), traverse(x.rhs, count, onFail))
        elif isinstance(x, ast.Or):
            if find(lambda x: isinstance(x, ast.Fail), x.lhs):
                return ast.Apply(ast.Lambda(None, traverse(x.lhs, 0, ast.Var)), traverse(x.rhs, count, onFail))
            else:
                return traverse(x.lhs, count, onFail)
##        elif isinstance(x, ast.Error):
##            # FIXME: If definition is recursive, this fails
##            outer = findIndex(lambda y: y == x.var, stack)
##
##            if outer is None:
##                return onFail(count + 1)
##            else:
##                return reduce(lambda exp, i: ast.Apply(exp, ast.Var(i)), range(x.arity, 0, -1), ast.Var(outer))
        elif isinstance(x, (ast.Fail, ast.Error)):
            return onFail(count + 1)
        elif isinstance(x, ast.Var):
            if isinstance(x.name, int):
                if x.name > count:
                    return ast.Var(x.name + 1)

            return x
        else:
            return x

    onFail = lambda _: ast.Apply(ast.Global('error'), ast.String('Pattern match failure'))
    return traverse(exp, 0, onFail)

def transformLet(bindings, exp):
    """let x = a and y = b and ... in ... → (λx.λy. ...) a b"""
    bindings = handlePatternMatching(bindings)
    (args, exps) = zip(*((binding.var, binding.exp) for binding in bindings))
    lhs = foldr(processLambda, exp, args)
    return transformCore(chainApply(lhs, *exps))

def transformLetRec(bindings, exp):
    """let rec x1 = a and x2 and ... = b in ... → Y* (λ x1 x2 ... xn. ...) (λ x1 x2 ... xn. a) (λ x1 x2 ... xn. b) ..."""
    return transformCore(makeRecScope(handlePatternMatching(bindings), exp))

def transformLetStar(bindings, exp):
    """let* x = a and y = b and ... in ... → let x = a in let y = b in ..."""
    return transformCore(foldr(lambda x, a: ast.Let([x], a), exp, bindings))

def transformBegin(exps):
    """begin exp_1; exp_2; ...; exp_n end → let rec ... in ..."""
    bindings = handlePatternMatching([exp for exp in exps if isinstance(exp, ast.Def)])
    body = [exp for exp in exps if not isinstance(exp, ast.Def)]
    return transformCore(makeRecScope(bindings, makeBody(body)))

def transformModule(exps):
    """Top level definitions."""
    bindings = handlePatternMatching([exp for exp in exps if isinstance(exp, ast.Def)])
    vars = chain(*(extractArgs(binding.var) for binding in bindings))
    labels = (makeLabel(ast.String(var.name), var) for var in vars)
    body = (exp for exp in exps if not isinstance(exp, ast.Def))
    return transformCore(makeRecScope(bindings, makeBody([*labels, *body])))

def transformLambda(arg, exp):
    x = processLambda(arg, exp)
    return ast.Lambda(transformCore(x.arg), transformCore(x.exp))

def transformAssign(lhs, rhs):
    if isinstance(lhs, (ast.List, ast.Pair, ast.Tuple)):
        # Bind value to temp variables then assign to target variables
        argList = extractArgs(lhs)
        tempArg = enumerateArgs(lhs, 'x', 1)
        body = makeBody([ast.Assign(arg, ast.Var('x{}'.format(i))) for (i, arg) in enumerate(argList, 1)])
        return ast.Apply(transformLambda(tempArg, body), transformCore(rhs))
    else:
        return ast.Assign(transformCore(lhs), transformCore(rhs))

def transformCore(x):
    """Core Language AST → Enriched Lambda Calculus AST"""
    if isinstance(x, ast.Grouping):
        return transformCore(x.exp)
    elif isinstance(x, ast.Apply):
        return ast.Apply(transformCore(x.lhs), transformCore(x.rhs))
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
    elif isinstance(x, ast.Module):
        return transformModule(x.exps)
    elif isinstance(x, ast.Begin):
        return transformBegin(x.exps)
    elif isinstance(x, ast.Assign):
        return transformAssign(x.lhs, x.rhs)
    elif isinstance(x, ast.BinOp):
        return chainApply(transformCore(x.f), transformCore(x.lhs), transformCore(x.rhs))
    elif isinstance(x, ast.UnOp):
        return chainApply(transformCore(x.f), transformCore(x.exp))
    elif isinstance(x, ast.Quantifier):
        return chainApply(transformCore(x.f), transformLambda(x.x, x.p), transformCore(x.xs))
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
    elif isinstance(x, ast.Global):
        return ls('Global', x.name)
    elif isinstance(x, ast.Var):
        return ls('Var', x.name)
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