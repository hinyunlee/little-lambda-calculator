from fractions import Fraction
import math

TRUE = True
FALSE = False
NIL = None

class Complex:
    def __init__(self, real=0, imaginary=0):
        """Complex fractions."""
        self.real = Fraction(real)
        self.imaginary = Fraction(imaginary)

    def __eq__(self, other):
        if isinstance(other, Complex):
            (u, v) = (other.real, other.imaginary)
        else:
            (u, v) = (other, 0)

        return self.real == u and self.imaginary == v

    def __abs__(self):
        return Fraction(math.sqrt(self.real**2 + self.imaginary**2))

    def __neg__(self):
        return Complex(-self.real, -self.imaginary)

    def __add__(self, other):
        if isinstance(other, Complex):
            (u, v) = (other.real, other.imaginary)
        else:
            (u, v) = (other, 0)

        return Complex(self.real + u, self.imaginary + v)

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
        """(u + v*i)/(x + y*i) = 1/(x**2 + y**2)*((u*x + v*y) + (v*x - u*y)*i)"""
        (u, v) = (self.real, self.imaginary)

        if isinstance(other, Complex):
            (x, y) = (other.real, other.imaginary)
        else:
            (x, y) = (other, 0)

        d = x**2 + y**2
        return Complex(Fraction(u*x + v*y, d), Fraction(v*x - u*y, d))

    def __pow__(self, other):
        (a, b) = (self.real, self.imaginary)

        if isinstance(other, Complex):
            (c, d) = (other.real, other.imaginary)
        else:
            (c, d) = (other, 0)

        rSquared = a**2 + b**2
        r = Fraction(math.sqrt(rSquared))
        theta = Fraction(math.atan2(a, b))
        return r**c*Fraction(math.exp(-d*theta))*Complex(math.cos(c*theta + Fraction(d*math.log(rSquared))/2), math.sin(c*theta + Fraction(d*math.log(rSquared))/2))

    def __rpow__(self, other):
        if isinstance(other, Complex):
            return other.__pow__(self)
        else:
            return Complex(other).__pow__(self)

    def __repr__(self):
        return '{}{}{}i'.format(self.real, '+' if self.imaginary >= 0 else '', self.imaginary)

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