from fractions import Fraction
import math

TRUE = True
FALSE = False
NIL = None

sgn = lambda x: -1 if x < 0 else 0 if x == 0 else 1

class Complex:
    def __init__(self, real=0, imaginary=0):
        """Complex number supporting arbitrary types.
        Might break if real or imaginary are complex numbers.
        """
        self.real = real
        self.imaginary = imaginary

    def __eq__(self, other):
        (x, y) = (self.real, self.imaginary)

        if isinstance(other, Complex):
            (u, v) = (other.real, other.imaginary)
        else:
            (u, v) = (other, 0)

        return x == u and y == v

    def __abs__(self):
        return math.sqrt(self.real**2 + self.imaginary**2)

    def __neg__(self):
        return Complex(-self.real, -self.imaginary)

    def __add__(self, other):
        (x, y) = (self.real, self.imaginary)

        if isinstance(other, Complex):
            (u, v) = (other.real, other.imaginary)
        else:
            (u, v) = (other, 0)

        return Complex(x + u, y + v)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other):
        """(x + y*i)*(u + v*i) = (x*u - y*v) + (x*v + y*u)*i"""
        (x, y) = (self.real, self.imaginary)
        
        if isinstance(other, Complex):
            (u, v) = (other.real, other.imaginary)
        else:
            (u, v) = (other, 0)

        return Complex(x*u - y*v, x*v + y*u)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """(u + v*i)/(x + y*i) = (1/(x**2 + y**2))*((u*x + v*y) + (v*x - u*y)*i)"""
        (u, v) = (self.real, self.imaginary)
        
        if isinstance(other, Complex):
            (x, y) = (other.real, other.imaginary)
        else:
            (x, y) = (other, 0)

        d = x**2 + y**2
        return Complex(Fraction(u*x + v*y, d), Fraction(v*x - u*y, d))

    def __pow__(self, n):
        # TODO: Complex exponents
        if isinstance(n, Complex):
            raise Exception('Not yet implemented')

        a = self.__abs__()**n
        b = math.atan2(self.real, self.imaginary)*n
        return Complex(a*math.cos(b), a*math.sin(b))

    def __repr__(self):
        if self.real == 0:
            return '{}i'.format(self.imaginary)
        else:
            sgnStr = lambda x: '+' if x >= 0 else '-'
            return '{}{}{}i'.format(self.real, sgnStr(self.imaginary), abs(self.imaginary))

    def __complex__(self):
        return complex(self.real, self.imaginary)

class Pair:
    def __init__(self, head, tail):
        self.head = head
        self.tail = tail

    def __repr__(self):
        p = self
        xs = []

        while isinstance(p, Pair):
            xs.append(repr(p.head))
            p = p.tail

        if p is not NIL:
            xs.append('.')
            xs.append(repr(p))

        return '[{}]'.format(' '.join(xs))

    def __eq__(self, other):
        if not isinstance(other, Pair):
            return False

        (x, y) = (self, other)

        while all(isinstance(a, Pair) for a in (x, y)):
            if x.head != y.head:
                return False

            (x, y) = (x.tail, y.tail)

        return x == y

    def __len__(self):
        return 2

    def __abs__(self):  # |self|
        i = 0
        current = self

        while isinstance(current, Pair):
            i = i + 1
            current = current.tail

        return i

    def __iter__(self):
        current = self

        while current:
            yield current
            current = current.tail

    def __getitem__(self, k):
        if k == 0:
            return self.head
        elif k == 1:
            return self.tail

    def __setitem__(self, k, v):
        if k == 0:
            self.head = v
        elif k == 1:
            self.tail = v

class Set:
    def __init__(self, xs):
        self.xs = list(set(xs))  # Unordered

    def __eq__(self, other):
        return isinstance(other, Set) and all(x in other.xs for x in self.xs)

    def __getitem__(self, k):
        return self.xs[k]

    def __setitem__(self, k, v):
        self.xs[k] = v
        self.xs = list(set(self.xs))

    def __iter__(self):
        return iter(self.xs)

    def __len__(self):
        return len(self.xs)

    def __repr__(self):
        return '{{{}}}'.format(', '.join(map(repr, self.xs)))

    # TODO