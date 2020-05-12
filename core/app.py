from .reader import Reader
from .evaluator import Evaluator
from .printer import Printer
from .data import NIL
from . import primitives
from traceback import print_exc
import sys
import argparse

def slurp(x):
    with open(x, encoding='utf-8') as f:
        return f.read()

def init():
    r = Reader()
    e = Evaluator(primitives.extend(), r.read(slurp('prelude')))
    p = Printer()
    return (r, e, p)

def repl():
    print('Little λ Calculator')
    (r, e, p) = init()

    while True:
        try:
            p.print(e.eval(r.read(input('> '))))
        except EOFError:
            sys.exit(0)
        except KeyboardInterrupt:
            pass
        except SystemExit:
            break
        except:
            print_exc()

def exec(x):
    (r, e, p) = init()

    try:
        p.print(e.eval(r.read(x)))
    except:
        print_exc()

def run():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('filename', nargs='?')
    args = parser.parse_args()

    if args.filename is not None:
        exec(slurp(args.filename))
    else:
        repl()
