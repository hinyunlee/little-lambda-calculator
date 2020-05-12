import unittest
from tests.test_base import TestBase
from core import data

class TestAdvanced(TestBase):
    def test_callcc(self):
        x = """
let
    f return := begin return 2; 3 end
in
    begin
        print (f (λx.x))
        print newline
        print (call/cc f)
        print newline
    end
        """
        self.readEvalPrint(x)

    def test_tco(self):
        x = """
let rec
    fact' n a := if n = 0 then a else fact' (n - 1) (n * a)
    fact n := fact' n 1
in
    fact 10000
        """
        self.readEvalPrint(x)

    def test_ycombinator(self):
        x = """
let
    Y := λf.(λx.f(x x))(λx.f(x x))
    fact f n := if n = 0 then 1 else n * f (n - 1)
in
    Y fact 10
        """
        result = self.readEvalPrint(x)
        self.assertEqual(result, 3628800)

        x = """
let
    Y* := λfs.Y(λf.map (λg.g f) fs)
    feven? [even? odd?] n := if n = 0 then true else odd? (n - 1)
    fodd? [even? odd?] n := if n = 0 then false else even? (n - 1)
in let
    [even? odd?] := Y*[feven? fodd?]
in
    even? 10
        """
        result = self.readEvalPrint(x)
        self.assertEqual(result, True)

    def test_quoting(self):
        x = """
let *
    quoted := reify((λx.λy. x ^ y) 2 3)
    unquoted := eval quoted
in
    begin
        print quoted
        print newline
        print unquoted
        print newline
    end
        """
        self.readEvalPrint(x)

    def test_mutual_recursion(self):
        x = """
let rec
    even? n := if n = 0 then true else odd? (n - 1)
    odd? n := if n = 0 then false else even? (n - 1)
in
    even? 10000
        """
        result = self.readEvalPrint(x)
        self.assertEqual(result, data.TRUE)

        x = """
even? n := if n = 0 then true else odd? (n - 1)
odd? n := if n = 0 then false else even? (n - 1)
odd? 10000
        """
        result = self.readEvalPrint(x)
        self.assertEqual(result, data.FALSE)

    def test_pattern_matching(self):
        x = """
let rec
    f 0 := 1
    f n := n * f (n - 1)
in
    f 10
        """
        result = self.readEvalPrint(x)
        self.assertEqual(result, 3628800)

        x = """
let rec
    fib 0 := 0
    fib 1 := 1
    fib n := fib(n - 1) + fib(n - 2)
in
    fib 10
        """
        result = self.readEvalPrint(x)
        self.assertEqual(result, 55)

    def test_destructuring(self):
        x = """
let
    ((x, y), z) := ((1, 2), 3)
in
    x + y + z
        """
        result = self.readEvalPrint(x)
        self.assertEqual(result, 6)

        x = """
let
    [x [y z]] := [1 [2 3]]
in
    x + y + z
        """
        result = self.readEvalPrint(x)
        self.assertEqual(result, 6)

        x = """
let
    [x (y, z)] := [1 (2, 3)]
in
    x + y + z
        """
        result = self.readEvalPrint(x)
        self.assertEqual(result, 6)

if __name__ == '__main__':
    unittest.main()
