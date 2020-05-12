from fractions import Fraction
import builtins
import math

class Printer:
    def format(self, x):
        if x is None:
            return ''
        elif x is True:
            return 'true'
        elif x is False:
            return 'false'
        elif isinstance(x, Fraction):
            return str(float(x))
        elif isinstance(x, (tuple, list)):
            return str(tuple(x))
        elif x == math.inf:
            return '∞'
        elif x == -math.inf:
            return '-∞'
        else:
            return str(x)

    def print(self, x):
        if x is not None:
            builtins.print(format(x))

# TODO