# Parser: (token index, left expression, operator, reserved words) -> (expression | None, next index)
# If left expression is None, context is NUD; otherwise context is LED
# Operator types: {infix, prefix, postfix, matchfix}
# Associativity: {l, r}
# NUD: {prefix, matchfix}
# LED: {infix, postfix}

# Atom ::= Identifier | Number | Symbol
# Term ::= Atom | MatchfixExp
# Exp ::= Term | Exp Exp | InfixExp | PrefixExp | PostfixExp
# InfixExp ::= Exp Op Exp ...
# PrefixExp ::= Op Exp ...
# PostfixExp ::= Exp Op ...
# MatchfixExp ::= Op(1) Exp Op(2)

from . import ast
from functools import partial, reduce
from fractions import Fraction
from .fp import foldr, foldr1
from .data import Complex

EMPTY_SET = frozenset()

class Parser:
    def __init__(self):
        self.tokens = None

        self.ops = {
            '(':     {'matchfix': {'parser': self.parseGrouped}},
            '[':     {'matchfix': {'parser': self.parseList}},
            '{':     {'matchfix': {'parser': self.parseSet}},
            '⌊':     {'matchfix': {'parser': partial(self.parseMatchfixOp, '⌊', '⌋', 'floor')}},
            '⌈':     {'matchfix': {'parser': partial(self.parseMatchfixOp, '⌈', '⌉', 'ceil')}},
            '|':     {'matchfix': {'parser': partial(self.parseMatchfixOp, '|', '|', '|_|')}},
            '↑':     {'infix':  {'precedence': 800, 'associativity': 'r', 'parser': partial(self.parseInfixOp, ['↑', '^'], '_↑_')}},
            '¬':     {'prefix': {'precedence': 700, 'associativity': 'r', 'parser': partial(self.parsePrefixOp, ['¬', 'not'], '¬_')}},
            '⋅':     {'infix':  {'precedence': 600, 'associativity': 'l', 'parser': partial(self.parseInfixOp, ['⋅', '×', '*'], '_⋅_')}},
            '/':     {'infix':  {'precedence': 600, 'associativity': 'l', 'parser': partial(self.parseInfixOp, ['/'], '_/_')}},
            'mod':   {'infix':  {'precedence': 600, 'associativity': 'l', 'parser': partial(self.parseInfixOp, ['mod'], '_mod_')}},
            '∧':     {'infix':  {'precedence': 600, 'associativity': 'l', 'parser': partial(self.parseInfixOp, ['∧', '&'], '_∧_')}},
            '∣':     {'infix':  {'precedence': 600, 'associativity': 'l', 'parser': partial(self.parseInfixOp, ['∣'], '_∣_')}},
            '+':     {'infix':  {'precedence': 500, 'associativity': 'l', 'parser': partial(self.parseInfixOp, ['+'], '_+_')}},
            '-':     {'infix':  {'precedence': 500, 'associativity': 'l', 'parser': partial(self.parseInfixOp, ['-'], '_-_')},
                      'prefix': {'precedence': 700, 'associativity': 'r', 'parser': partial(self.parsePrefixOp, ['-'], '-_')}},
            '∨':     {'infix':  {'precedence': 500, 'associativity': 'l', 'parser': partial(self.parseInfixOp, ['∨', 'or'], '_∨_')}},
            '≤':     {'infix':  {'precedence': 400, 'associativity': 'l', 'parser': self.parseComparison}},
            '<':     {'infix':  {'precedence': 400, 'associativity': 'l', 'parser': self.parseComparison}},
            '≥':     {'infix':  {'precedence': 400, 'associativity': 'l', 'parser': self.parseComparison}},
            '>':     {'infix':  {'precedence': 400, 'associativity': 'l', 'parser': self.parseComparison}},
            '=':     {'infix':  {'precedence': 400, 'associativity': 'l', 'parser': self.parseComparison}},
            '≠':     {'infix':  {'precedence': 400, 'associativity': 'l', 'parser': self.parseComparison}},
            '⇒':     {'infix':  {'precedence': 400, 'associativity': 'l', 'parser': partial(self.parseInfixOp, ['⇒', '=>'], '_⇒_')}},
            '⇐':     {'infix':  {'precedence': 400, 'associativity': 'l', 'parser': partial(self.parseInfixOp, ['⇐'], '_⇐_')}},
            '⇔':     {'infix':  {'precedence': 400, 'associativity': 'l', 'parser': partial(self.parseInfixOp, ['⇔', '<=>'], '_⇔_')}},
            '∈':     {'infix':  {'precedence': 400, 'associativity': 'l', 'parser': partial(self.parseInfixOp, ['∈', 'in'], '_∈_')}},
            '∉':     {'infix':  {'precedence': 400, 'associativity': 'l', 'parser': partial(self.parseInfixOp, ['∉', 'not-in'], '_∉_')}},
            '::':    {'infix':  {'precedence': 300, 'associativity': 'r', 'parser': partial(self.parseInfixOp, ['::'], '_::_')}},
            '||':    {'infix':  {'precedence': 300, 'associativity': 'r', 'parser': partial(self.parseInfixOp, ['||'], '_||_')}},
            '∘':     {'infix':  {'precedence': 200, 'associativity': 'r', 'parser': partial(self.parseInfixOp, ['∘', '.'], '_∘_')}},
            'λ':     {'prefix': {'precedence': 100, 'associativity': 'r', 'parser': self.parseLambda}},
            '∀':     {'prefix': {'precedence': 100, 'associativity': 'r', 'parser': partial(self.parseQuantifier, '∀', 'for-all')}},
            '∃':     {'prefix': {'precedence': 100, 'associativity': 'r', 'parser': partial(self.parseQuantifier, '∃', 'exists')}},
            '←':     {'infix': {'precedence': 100, 'associativity': 'r', 'parser': self.parseAssign}},
            'if':    {'infix':  {'precedence': 100, 'associativity': 'r', 'parser': self.parseIf},
                      'prefix': {'precedence': 100, 'associativity': 'r', 'parser': self.parseIf}},
            'let':   {'prefix': {'precedence': 100, 'associativity': 'r', 'parser': self.parseLet}},
            'begin': {'prefix': {'precedence': 100, 'associativity': 'r', 'parser': self.parseBegin}},
            'where': {'infix':  {'precedence': 0, 'associativity': 'r', 'parser': self.parseWhere}}
        }

        self.opAliases = {
            'not': '¬',
            '^': '↑',
            '×': '⋅',
            '*': '⋅',
            '<=': '≤',
            '>=': '≥',
            '/=': '≠',
            '=>': '⇒',
            '<=>': '⇔',
            '&': '∧',
            'or': '∨',
            '.': '∘',
            '\\': 'λ',
            'lambda': 'λ',
            '<-': '←',
            'in': '∈',
            'not-in': '∉'
        }

        self.constants = {
            'nil': ast.NIL,
            'true': ast.TRUE,
            'false':ast.FALSE
        }

    def hasNextToken(self, i):
        return i < len(self.tokens)

    def isEOF(self, i):
        return i >= len(self.tokens)

    def failIfEOF(self, i):
        if self.isEOF(i):
            raise Exception('Unexpected EOF')

    def getNextToken(self, i):
        self.failIfEOF(i)
        return (self.tokens[i], i + 1)

    def peekToken(self, i):
        if self.isEOF(i): return (None, i)
        return self.tokens[i]

    def isOperator(self, t):
        return any(t.value in xs for xs in (self.ops, self.opAliases))

    def getOperator(self, t):
        return self.ops[self.opAliases.get(t.value, t.value)]

    def peek(self, fs, i, lhs=None, op=None, rws=EMPTY_SET):
        if self.isEOF(i): return (None, i)

        if callable(fs):
            fs = [fs]

        rs = (f(i, lhs, op, rws) for f in fs)
        return next(((x, i) for (x, i) in rs if x is not None), (None, i))

    def consume(self, fs, i, lhs=None, op=None, rws=EMPTY_SET):
        self.failIfEOF(i)

        if callable(fs):
            fs = [fs]

        rs = (f(i, lhs, op, rws) for f in fs)
        (x, i) = next(((x, i) for (x, i) in rs if x is not None), (None, i))

        if x is None:
            (token, _) = self.getNextToken(i)
            raise Exception('Unexpected token', token.value, token.pos)

        return (x, i)

    def repeat(self, fs, i, lhs=None, op=None, rws=EMPTY_SET):
        exps = []
        (exp, i) = self.peek(fs, i, lhs, op, rws)

        while exp is not None:
            exps.append(exp)
            (exp, i) = self.peek(fs, i, lhs, op, rws)

        return (exps, i)

    def parseKeyword(self, kws, i, lhs=None, op=None, rws=EMPTY_SET):
        """Parse a keyword. If a list of keywords is given, parse any one of them."""
        if self.isEOF(i): return (None, i)

        if isinstance(kws, str):
            kws = [kws]

        (t1, j) = self.getNextToken(i)

        if t1.value == '\n':
            k = self.skipNewlines(j)
            if self.isEOF(k): return (None, i)
            (t2, k) = self.getNextToken(k)
            if t2.value in kws: return (t2.value, k)

        if t1.value in kws:
            return (t1.value, j)
        else:
            return (None, i)

    def consumeKeyword(self, keyword, i, lhs=None, op=None, rws=EMPTY_SET):
        return self.consume(partial(self.parseKeyword, keyword), i)

    def parseNewline(self, i, lhs=None, op=None, rws=EMPTY_SET):
        if self.isEOF(i): return (None, i)
        (token, j) = self.getNextToken(i)

        if token.name == 'Newline':
            return (token.value, j)
        else:
            return (None, i)

    def skipNewlines(self, i):
        while self.hasNextToken(i) and self.peekToken(i).name == 'Newline':
            i = i + 1

        return i

    def parseAtom(self, i, lhs=None, op=None, rws=EMPTY_SET):
        self.failIfEOF(i)
        (token, j) = self.getNextToken(i)
        name = token.name
        value = token.value

        if name == 'Number':
            def parseNumber(value):
                if value[-1] in ('i', 'j'):
                    return Complex(0, parseNumber(value[:-1]))
                elif value[-1] in ('f', 'F'):
                    return float(value[:-1])
                elif any(x in value for x in ('.', 'e', 'E')):
                    return Fraction(value)
                else:
                    return int(value)

            return (ast.Number(parseNumber(value)), j)
        elif name == 'String':
            q = value[0]
            return (ast.String(value[1:-1].replace(q*2, q)), j)
        else:
            return (self.constants.get(value) or ast.Var(value), j)

    def parseLambda(self, i, lhs, op=None, rws=EMPTY_SET):
        """'λ' Term* '.' Exp"""
        (_, i)    = self.consumeKeyword(('λ', 'lambda', '\\'), i)
        (args, i) = self.repeat(self.parseTerm, i, None, None, frozenset(['.']))
        (_, i)    = self.consumeKeyword('.', i)
        (body, i) = self.consumeExp(i, None, op, rws)

        if len(args) == 0:
            raise Exception('Missing argument in lambda', i)
        else:
            return (foldr(lambda arg, rhs: ast.Lambda(arg, rhs), body, args), i)

    def parseDef(self, i, lhs, op=None, rws=EMPTY_SET):
        (_, i)   = self.consumeKeyword((':=', '≔'), i)
        (exp, i) = self.consumeExp(i, None, op, rws)
        return (ast.Def(lhs, exp), i)

    def parseBindings(self, seps, i, lhs, op=None, rws=EMPTY_SET):
        """Exp ':=' Exp (seps Exp ':=' Exp)*"""
        def parseBinding(i, lhs, op=None, rws=EMPTY_SET):
            """Exp ':=' Exp"""
            (lhs, i) = self.parseExp(i, None, None, rws | frozenset((':=', '≔')))
            if lhs is None: return (None, i)
            return self.parseDef(i, lhs, op, rws)

        def parseAndBinding(seps, i, lhs, op=None, rws=EMPTY_SET):
            """seps Exp ':=' Exp"""
            (kw, i) = self.parseKeyword(seps, i)
            if kw is None: return (None, i)
            return parseBinding(i, None, op, rws)

        rws = rws | frozenset(seps)
        (b, i)  = parseBinding(i, None, op, rws)
        (bs, i) = self.repeat(partial(parseAndBinding, seps), i, None, op, rws)
        return ([b, *bs], i)

    def getScopeAST(self, t):
        if t is None:
            return ast.Let
        elif t == 'rec':
            return ast.LetRec
##        elif t == 'nonrec':
##            return ast.Let
        elif t == '*':
            return ast.LetStar
        else:
            raise Exception('Unknown scope modifier', t)

    def parseLet(self, i, lhs, op=None, rws=EMPTY_SET):
        """'let' ('rec' | '*')? Exp (':=' | '=') Exp (('and' | ',' | ';' | NL) Exp (':=' | '=') Exp)* 'in' Exp"""
        (_, i)   = self.consumeKeyword('let', i)
        (t, i)   = self.parseKeyword(('rec', '*'), i)
        (bs, i)  = self.parseBindings(('and', ',', ';', '\n'), i, None, None, frozenset(['in']))
        (_, i)   = self.consumeKeyword('in', i)
        (exp, i) = self.consumeExp(i, None, op, rws)
        return (self.getScopeAST(t)(bs, exp), i)

    def parseWhere(self, i, lhs, op=None, rws=EMPTY_SET):
        """lhs 'where' ('rec' | '*')? Exp (':=' | '=') Exp (('and' | ',') Exp (':=' | '=') Exp)*"""
        (_, i)  = self.consumeKeyword('where', i)
        (t, i)  = self.parseKeyword(('rec', '*'), i)
        (bs, i) = self.parseBindings(('and', ','), i, None, op, rws)
        return (self.getScopeAST(t)(bs, lhs), i)

    def parseIf(self, i, lhs, op=None, rws=EMPTY_SET):
        """'if' Exp 'then' Exp ('else' Exp)? | lhs 'if' Exp ('else' Exp)?"""
        (_, i) = self.consumeKeyword('if', i)

        if lhs is None:
            (p, i)   = self.consumeExp(i, None, None, frozenset(['then']))
            (_, i)   = self.consumeKeyword('then', i)
            (lhs, i) = self.consumeExp(i, None, op, rws | frozenset(['else']))
        else:
            (p, i) = self.consumeExp(i, None, op, rws | frozenset(['else']))

        (kw, i) = self.parseKeyword('else', i)
        if kw is None: return (ast.If(p, lhs, ast.FAIL), i)
        (rhs, i) = self.consumeExp(i, None, op, rws)
        return (ast.If(p, lhs, rhs), i)

    def parseExpList(self, seps, i, lhs, op=None, rws=EMPTY_SET):
        """Exp (seps Exp)*"""
        def parseNextExp(sep, i, lhs, op=None, rws=EMPTY_SET):
            """seps Exp"""
            (kw, i) = self.parseKeyword(sep, i)
            if kw is None: return (None, i)
            return self.consumeExp(i, None, op, rws)

        rws = rws | frozenset(seps)
        (exp, i) = self.parseExp(i, None, None, rws)
        if exp is None: return ([], i)
        (sep, _) = self.parseKeyword(seps, i)  # Must be separated by the same symbol
        if sep is None: return ([exp], i)
        (exps, i) = self.repeat(partial(parseNextExp, sep), i, None, None, rws | frozenset([sep]))
        return ([exp, *exps], i)

    def parseGrouped(self, i, lhs, op=None, rws=EMPTY_SET):
        """'(' (Exp ((',' Exp)* | (';' Exp)*))? ')'"""
        (_, i)    = self.consumeKeyword('(', i)
        (exps, i) = self.parseExpList((',', ';'), i, None, None, frozenset([')']))
        (_, i)    = self.consumeKeyword(')', i)

        if len(exps) == 1:
            return (ast.Grouping(exps[0]), i)
        else:
            return (ast.Tuple(exps), i)

    def parseList(self, i, lhs, op=None, rws=EMPTY_SET):
        """'[' Term* ']'"""
        (_, i)    = self.consumeKeyword('[', i)
        (exps, i) = self.repeat(self.parseTerm, i, None, None, frozenset([']', '.']))
        (dot, i)  = self.parseKeyword('.', i)

        if dot is None:
            tail = ast.NIL  # []
        else:
            (tail, i) = self.consume(self.parseTerm, i, None, None, frozenset([']']))

        (_, i) = self.consumeKeyword(']', i)
        return (ast.List(exps, tail), i)

    def parseSet(self, i, lhs, op=None, rws=EMPTY_SET):
        """'{' (Exp (',' Exp)*)? '}'"""
        (_, i)    = self.consumeKeyword('{', i)
        (exps, i) = self.parseExpList(',', i, None, None, frozenset(['}']))
        (_, i)    = self.consumeKeyword('}', i)
        return (ast.Set(exps), i)

    def parseBlock(self, seps, i, lhs=None, op=None, rws=EMPTY_SET):
        """Exp (':=' Exp)? (seps Exp (':=' Exp)?)*"""
        def parseLine(i, lhs, op=None, rws=EMPTY_SET):
            """Exp (':=' Exp)?"""
            seps = (':=', '≔')
            (lhs, i) = self.parseExp(i, None, op, rws | frozenset(seps))
            if lhs is None: return (lhs, i)
            (kw, _) = self.parseKeyword(seps, i)
            if kw is None: return (lhs, i)
            return self.parseDef(i, lhs, op, rws)

        def parseNextLine(i, lhs, op=None, rws=EMPTY_SET):
            """seps Exp (':=' Exp)?"""
            (kw, i) = self.parseKeyword(seps, i)
            if kw is None: return (None, i)
            return parseLine(i, None, op, rws)

        rws = rws | frozenset(seps)
        (exp, i) = parseLine(i, None, op, rws)
        if exp is None: return ([], i)
        (exps, i) = self.repeat(parseNextLine, i, None, op, rws)
        return ([exp, *exps], i)

    def parseBegin(self, i, lhs, op=None, rws=EMPTY_SET):
        """'begin' Exp (':=' Exp)? ((',' | ';' | NL) Exp (':=' Exp)?)* 'end'"""
        (_, i)    = self.consumeKeyword('begin', i)
        (exps, i) = self.parseBlock((',', ';', '\n'), i, None, None, frozenset(['end']))
        (_, i)    = self.consumeKeyword('end', i)
        return (ast.Begin(exps), i)

    def parsePrefixOp(self, kws, f, i, lhs, op=None, rws=EMPTY_SET):
        """kws Exp"""
        (_, i)   = self.consumeKeyword(kws, i)
        (exp, i) = self.consumeExp(i, None, op, rws)
        return (ast.PrefixOp(ast.Var(f), exp), i)

    def parseInfixOp(self, kws, f, i, lhs, op=None, rws=EMPTY_SET):
        """lhs kws Exp"""
        (_, i)   = self.consumeKeyword(kws, i)
        (rhs, i) = self.consumeExp(i, None, op, rws)
        return (ast.BinOp(ast.Var(f), lhs, rhs), i)

    def parsePostfixOp(self, kws, f, i, lhs, op=None, rws=EMPTY_SET):
        """lhs kws"""
        if lhs is None:
            (token, i) = self.getNextToken(i)
            raise Exception('Missing left hand side expression', token.name, token.value)
        else:
            (_, i) = self.consumeKeyword(kws, i)
            return (ast.PostfixOp(ast.Var(f), lhs), i)

    def parseMatchfixOp(self, kw1, kw2, f, i, lhs, op=None, rws=EMPTY_SET):
        """kw1 Exp? kw2"""
        (_, i)   = self.consumeKeyword(kw1, i)
        (exp, i) = self.parseExp(i, None, None, frozenset([kw2]))
        (_, i)   = self.consumeKeyword(kw2, i)
        return (ast.MatchfixOp(ast.Var(f), exp or ast.NIL), i)

    def parseQuantifier(self, kw, f, i, lhs, op=None, rws=EMPTY_SET):
        """kw x ∈ A p(x)"""
        (_, i)  = self.consumeKeyword(kw, i)
        (x, i)  = self.parseTerm(i)
        (_, i)  = self.consumeKeyword(('∈', 'in'), i)
        (xs, i) = self.parseTerm(i)
        (p, i)  = self.consumeExp(i, None, op, rws)
        return (ast.Quantifier(ast.Var(f), x, xs, p), i)

    def parseAssign(self, i, lhs, op=None, rws=EMPTY_SET):
        """lhs ← Exp"""
        (_, i)   = self.consumeKeyword(('←', '<-'), i)
        (exp, i) = self.consumeExp(i, None, op, rws)
        return (ast.Assign(lhs, exp), i)

    def parseComparison(self, i, lhs, op=None, rws=EMPTY_SET):
        """lhs (< | <= | > | >= | = | /=) Exp ((< | <= | > | >= | = | /=) Exp)*

        x1 op1 x2 → x1 op1 x2
        x1 op1 x2 op2 x3 ... → (λy1 y2 y3 ... . (y1 op1 y2) and (y2 op2 y3) and ...) x1 x2 x3
        Note: y1, y2, ..., yn are memoized.
        """
        def parseRest(i, lhs, op=None, rws=EMPTY_SET):
            """op Exp"""
            ops = ('<', '≤', '<=', '=<', '>', '≥', '>=', '=', '≠', '/=')
            rws = rws | frozenset(ops)
            (kw, i)  = self.parseKeyword(ops, i)
            if kw is None: return (None, i)
            (exp, i) = self.consumeExp(i, None, op, rws)
            return ((kw, exp), i)

        varFromKw = lambda kw: ast.Var('_{}_'.format(kw))
        (xs, i) = self.repeat(parseRest, i, lhs, op, rws)

        if len(xs) == 1:
            (kw, rhs) = xs[0]
            return (ast.BinOp(varFromKw(self.opAliases.get(kw, kw)), lhs, rhs), i)
        else:
            (kws, exps) = zip(*xs)
            ops = [varFromKw(self.opAliases.get(kw, kw)) for kw in kws]
            exps = [lhs, *exps]  # len(exps) = len(ops) + 1
            args = [ast.Var('x' + str(i)) for i in range(len(exps))]
            binOps = [ast.BinOp(op, args[i], args[i + 1]) for (i, op) in enumerate(ops)]
            andVar = ast.Var('_∧_')
            body = foldr1(lambda lhs, rhs: ast.BinOp(andVar, lhs, rhs), binOps)
            lhs = foldr(ast.Lambda, body, args)
            return (reduce(lambda lhs, rhs: ast.Apply(lhs, rhs), exps, ast.Lambda(args, body)), i)

    def parseTerm(self, i, lhs=None, op=None, rws=EMPTY_SET):
        """Term"""
        if self.isEOF(i): return (None, i)
        token = self.peekToken(i)
        if token.value in rws: return (None, i)  # May check for newline
        i = self.skipNewlines(i)
        if self.isEOF(i): return (None, i)
        token = self.peekToken(i)
        if token.value in rws: return (None, i)  # Without newline

        if self.isOperator(token):
            opBase = self.getOperator(token)

            if 'matchfix' in opBase:
                nextOp = opBase['matchfix']
                return nextOp['parser'](i, None, nextOp, rws)
            else:
                raise Exception('Unexpected operator', token.name, token.value)
        else:
            return self.parseAtom(i)

    def parseLHS(self, i, lhs=None, op=None, rws=EMPTY_SET):
        i = self.skipNewlines(i)
        if self.isEOF(i): return (None, i)
        token = self.peekToken(i)
        if token.value in rws: return (None, i)  # Without newline

        if self.isOperator(token):
            opBase = self.getOperator(token)
            opType = next((x for x in ('prefix', 'matchfix') if x in opBase), None)

            if opType:
                nextOp = opBase[opType]
                return nextOp['parser'](i, None, nextOp, rws)
            else:
                raise Exception('Unexpected operator', token.value, token.pos)
        else:
            return self.parseAtom(i)

    def shouldSkipNewlines(self, i, rws=EMPTY_SET):
        """Skip if next line starts with a binary-only operator or a reserved word."""
        i = self.skipNewlines(i)
        if self.isEOF(i): return False
        token = self.peekToken(i)
        return self.isOperator(token) and all(x not in self.getOperator(token) for x in ('prefix', 'matchfix')) and token.value not in rws

    def parseExp(self, i, lhs=None, op=None, rws=EMPTY_SET):
        """Exp"""
        (lhs, i) = self.parseLHS(i, None, op, rws)

        while self.hasNextToken(i):
            if self.shouldSkipNewlines(i, rws):
                i = self.skipNewlines(i)
                token = self.peekToken(i)
            else:
                token = self.peekToken(i)
                if token.value in rws: return (lhs, i)  # May check for newline
                i = self.skipNewlines(i)
                if self.isEOF(i): return (lhs, i)
                token = self.peekToken(i)
                if token.value in rws: return (lhs, i)  # Without newline

            if self.isOperator(token):
                opBase = self.getOperator(token)
                opType = next((x for x in ('infix', 'postfix', 'matchfix') if x in opBase), None)  # Order matters

                if opType in ('infix', 'postfix'):
                    nextOp = opBase[opType]

                    if op is None or op['precedence'] < nextOp['precedence'] or op['precedence'] == nextOp['precedence'] and nextOp['associativity'] == 'r':
                        (lhs, i) = nextOp['parser'](i, lhs, nextOp, rws)
                    else:
                        return (lhs, i)
                elif opType == 'matchfix':
                    nextOp = opBase[opType]
                    (rhs, i) = nextOp['parser'](i, None, nextOp, rws)
                    lhs = ast.Apply(lhs, rhs)
                else:
                    raise Exception('Unexpected operator', token.value, token.pos)
            else:
                (rhs, i) = self.consume(self.parseTerm, i)
                lhs = ast.Apply(lhs, rhs)

        return (lhs, i)

    def consumeExp(self, i, lhs=None, op=None, rws=EMPTY_SET):
        return self.consume(self.parseExp, i, None, op, rws)

    def parse(self, tokens):
        self.tokens = tokens  # FIXME: Ugly
        (exps, i) = self.parseBlock((',', ';', '\n'), 0)

        if self.hasNextToken(i):
            (token, _) = self.getNextToken(i)
            self.tokens = None  # FIXME: Ugly
            raise Exception('Unexpected token', token.name, token.value)
        else:
            self.tokens = None  # FIXME: Ugly
            return ast.Module(exps)
