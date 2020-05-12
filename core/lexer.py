import re
import sys
from functools import reduce, partial
from itertools import dropwhile, groupby, chain

class Token:
    def __init__(self, name, value=None, start=None, end=None, pos=None):
        self.name = name
        self.value = value
        self.start = start
        self.end = end
        self.pos = pos  # (line number: [1, n], column number: [1, n])

    def __repr__(self):
        if self.value is None or len(self.value) == 0:
            return "[{} {}]".format(self.name, self.pos)
        else:
            return "[{} {} {}]".format(self.name, repr(self.value), self.pos)

genLineLengths = lambda s: tuple(map(len, s.split('\n')))

def matchNestedGrouping(s, lpat, ipat, rpat, fi, i):
    """Match nested grouping and return (start index, end index) if a match is found.
    Returns None if match failed.
    lpat -- Opening pattern.
    ipat -- Inner opening pattern.
    rpat -- Closing pattern.
    fi -- Inner matcher.
    """
    firstMatch = lpat.match(s, i)
    j = firstMatch.end()
    innerMatch = ipat.search(s, j)
    lastMatch = rpat.search(s, j)

    while innerMatch or lastMatch:
        if innerMatch:
            j = fi(innerMatch)
            innerMatch = ipat.search(s, j)
            lastMatch = rpat.search(s, j)
        elif lastMatch:
            return (i, lastMatch.end())
    
    return None

def matchBlockComment(s, o, c, i):
    lpat = re.compile(o)
    rpat = re.compile(c)
    fi = lambda innerMatch: matchBlockComment(s, o, c, innerMatch.start())[1]
    m = matchNestedGrouping(s, lpat, lpat, rpat, fi, i)

    if m:
        return m
    else:
        raise Exception('Unmatched block comment', o, c, findDocPos(s, i))

def processBlockComment(o, c, s, i, m):
    end = matchBlockComment(s, o, c, i)[1]
    return Token('BlockComment', s[i:end], i, end)

def matchString(s, q, i):
    lpat = re.compile(q)
    ipat = re.compile(q + q)  # Double-quoting escapes a quote
    fi = lambda innerMatch: innerMatch.end()
    m = matchNestedGrouping(s, lpat, ipat, lpat, fi, i)

    if m:
        return m
    else:
        raise Exception('Unmatched string', q, findDocPos(s, i))

def processString(s, i, m):
    end = matchString(s, m.group(), i)[1]
    return Token('String', s[i:end], i, end)

def findDocPos(s, i):
    """Get the position (line number, column number) of the char at index i."""
    lineLengths = genLineLengths(s)
    lineIndex = 0
    lineStart = 0
    lineEnd = lineStart + lineLengths[lineIndex] + 1

    while i >= lineEnd:
        lineIndex = lineIndex + 1
        lineStart = lineEnd
        lineEnd = lineStart + lineLengths[lineIndex] + 1

    return (lineIndex + 1, i - lineStart + 1)

def resolveTokenPos(s, tokens):
    """Attach position (line number, column number) to each token."""
    if len(tokens) == 0:
        return tokens

    ts = []
    lineLengths = genLineLengths(s)
    lineIndex = 0
    lineStart = 0
    lineEnd = lineLengths[lineIndex] + 1

    for t in tokens:
        while t.start >= lineEnd:  # Assuming that t[i].start <= t[i + 1].start
            lineIndex = lineIndex + 1
            lineStart = lineEnd
            lineEnd = lineStart + lineLengths[lineIndex] + 1

        pos = (lineIndex + 1, t.start - lineStart + 1)
        ts.append(Token(t.name, t.value, t.start, t.end, pos))

    return ts

class Lexer:
    def __init__(self):
        lineComment  = '#'
        blockComment = [r'#\|', r'\|#']
        number       = r'([0-9]+)(\.[0-9]+)?([eE][+-]?[0-9]+)?([ij])?[fF]?'
        quotes       = ['"', "'"]
        newlines     = [r'\n', r'\r', r'\f']
        spaces       = [' ', r'\t', r'\v']
        separators   = [r'\(', r'\)', r'\[', r'\]', r'\{', r'\}',
                        r'\.', ',', ';', ':',
                        '#', r'\|',
                        '⌊', '⌋', '⌈', '⌉',
                        '::', r'\|\|']
        operators    = [r'\+', '-', '⋅', '×', r'\*', '/', '↑', r'\^', '∣',
                        '<', '≤', '<=', '=<', '>', '≥', '>=', '=', '≠', '/=',
                        '∧', '&', '∨',
                        '←', '<-', ':=', '≔',
                        '⇒', '=>', '⇐', '⇔', '<=>',
                        '∈', '∉',
                        '∘',
                        '::', r'\|\|',
                        '∀', '∃',
                        'λ', r'\\',]

        sortLongestFirst = lambda xs: sorted(xs, key=len, reverse=True)

        lcRe  = re.compile(r'{}[^{}]*'.format(lineComment, ''.join(newlines)))
        bcRe  = re.compile(blockComment[0])
        strRe = re.compile('{}'.format('|'.join(quotes)))
        spRe  = re.compile('({})+'.format('|'.join(spaces)))
        nlRe  = re.compile('({})'.format('|'.join(newlines)))
        sepRe = re.compile('({})'.format('|'.join(sortLongestFirst(separators))))
        opRe  = re.compile('_?({0})(_({0})?)?'.format('|'.join(sortLongestFirst(operators))))
        numRe = re.compile(number)
        idRe  = re.compile('[^' + ''.join(spaces + newlines + separators) + ']+')

        tokenize = lambda name: lambda s, i, m: Token(name, m.group(), i, m.end())

        self.rules = [
            (bcRe.match,  partial(processBlockComment, *blockComment)),
            (lcRe.match,  tokenize('LineComment')),
            (spRe.match,  tokenize('Space')),
            (nlRe.match,  tokenize('Newline')),
            (numRe.match, tokenize('Number')),
            (strRe.match, processString),
            (opRe.match,  tokenize('Operator')),
            (sepRe.match, tokenize('Separator')),
            (idRe.search, tokenize('Identifier'))
        ]

        notTokens = lambda xs: lambda x: x.name not in xs

        def trimTokens(xs):
            nextIndex = lambda ts: next((i for i, t in enumerate(ts) if t.name not in xs), 0)
            return lambda ts: ts[nextIndex(ts):len(ts) - nextIndex(reversed(ts))]

        def dedupeConsecutiveTokens(xs):
            filterWithAcc = lambda f, xs: reduce(lambda a, x: f(a, x) and a.append(x) or a, xs, [])
            return lambda ts: filterWithAcc(lambda ys, t: t.name not in xs or not ys or ys[-1].name not in xs, ts)

        nl = ['Newline']

        self.preprocess = [
            partial(re.sub, r'^([ \t\v]*[\n\r\f]+)+', ''),  # Remove leading whitespace
            partial(re.sub, r'([\n\r\f]+[ \t\v]*)+$', ''),  # Remove trailing whitespace
            partial(re.sub, r'^[ \t\v\n\r\f]+$', '')
        ]

        self.filters = [notTokens(['LineComment', 'BlockComment', 'Space'])]        
        self.postprocess = [trimTokens(nl), dedupeConsecutiveTokens(nl)]

    def lex(self, s):
        def step(s, i):
            matches = ((f(s,  i), g) for f, g in self.rules)
            return next((g(s, i, m) for m, g in matches if m), None)

        s = reduce(lambda s, f: f(s), self.preprocess, s)
        i = 0
        n = len(s)
        tokens = []

        while i < n:
            token = step(s, i)

            if token:
                if all(f(token) for f in self.filters):
                    tokens.append(token)

                i = token.end
            else:
                raise Exception('Failed to read', s[i], findDocPos(s, i))

        return resolveTokenPos(s, tuple(reduce(lambda a, f: f(a), self.postprocess, tokens)))
