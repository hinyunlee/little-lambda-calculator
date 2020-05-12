# Functional programming library

from functools import reduce

flip = lambda f: lambda x, y: f(y, x)
foldl = lambda f, a, xs: reduce(f, xs, a)
foldl1 = lambda f, xs: reduce(f, xs)
foldr = lambda f, a, xs: reduce(flip(f), reversed(xs), a)
foldr1 = lambda f, xs: reduce(flip(f), reversed(xs))
unnest = lambda xss: [x for xs in xss for x in xs]
chain = lambda f, xs: [x for xs in [f(x) for x in xs] for x in xs]

def tap(f):
    def g(x):
        f(x)
        return x

    return g
