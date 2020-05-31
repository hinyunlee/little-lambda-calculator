from fractions import Fraction
import builtins
import math
from .data import Complex, Pair

class Printer:
    def format(self, x):
        if x is None:
            return ''
        elif x is True:
            return 'true'
        elif x is False:
            return 'false'
        elif isinstance(x, Fraction):
            if x.denominator == 1:
                return str(x)
            else:
                return str(float(x))
        elif isinstance(x, (tuple, list)):
            return '({})'.format(', '.join(map(self.format, x)))
        elif isinstance(x, Pair):
            xs = []

            while isinstance(x, Pair):
                xs.append(x.head)
                x = x.tail

            a = ' '.join(map(self.format, xs))
            b = ' . {}'.format(x) if x is not None else ''
            return '[{}{}]'.format(a, b)
        elif isinstance(x, Complex):
            if x.imaginary == 0:
                return self.format(x.real)
            elif x.real == 0:
                return '{}i'.format(x.imaginary)
            else:
                a = self.format(x.real)
                b = ('+' if x.imaginary >= 0 else '') + self.format(x.imaginary)
                return '{}{}i'.format(a, b)
        elif x == math.inf:
            return '∞'
        elif x == -math.inf:
            return '-∞'
        else:
            return str(x)

    def print(self, x):
        if x is not None:
            builtins.print(self.format(x))