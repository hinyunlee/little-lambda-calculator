import unittest
from tests.test_base import TestBase
from fractions import Fraction
from core import data

# TODO

class TestBasic(TestBase):
    def test_arithmetic(self):
        r = self.readEvalPrint
        self.assertEqual(r('1'), 1)
        self.assertEqual(r('2 + 2'), 4)
        self.assertEqual(r('1 + 2 * 3'), 7)
        self.assertEqual(r('(1 + 2) * 3'), 9)
        self.assertEqual(r('1 * (2 + 3)'), 5)
        self.assertEqual(r('1 + 2 - 3 * 4 / 5'), Fraction('0.6'))
        self.assertEqual(r('1 / 2 / 2'), Fraction(1, 4))
        self.assertEqual(r('0.03 * 0.03 * 0.03'), Fraction('2.7e-05'))
        self.assertEqual(r('2 ^ 2 ^ 2'), 16)
        self.assertEqual(r('1+1i'), data.Complex(1, 1))
        self.assertEqual(r('1+2i + 3+4i'), data.Complex(4, 6))
        self.assertEqual(r('1i ^ 1i'), 1)

    def test_abstraction(self):
        xs = [
            'λx.x',
            'λx.x x',
            'λx.x * 2',
            'let x := 2 in x + x',
            'let x := 2 in let y := 3 in x + y',
            'let x := 2 and y := 3 in x + y',
            'let x := (λx.x) ((λx.x x) (λx.x)) (λx.x) 2 in x + x',
            '(λx.x)(λy.y)',
            '(λx.x x)(λx.x)',
            '((λx.x)(λx.x))(λx.x)',
            '(λx y z. x + y + z) 1 2 3',
            'x + x where x := 2',
            'x + x where x := y ^ y where y := 2',
            'let x := 1 :: 2 :: 3 :: nil in head x',
            'let x := 1 :: 2 :: 3 :: nil in head (tail (tail x))'
        ]

        for x in xs:
            self.readEvalPrint(x)

    def test_condition(self):
        r = self.readEvalPrint
        self.assertEqual(r('if true then 1 else 0'), 1)
        self.assertEqual(r('1 if true else 0'), 1)
        self.assertEqual(r('if false then 1 else 0'), 0)
        self.assertEqual(r('1 if false else 0'), 0)
        self.assertEqual(r('if nil then 1 else 0'), 0)
        self.assertEqual(r('1 if nil else 0'), 0)
        self.assertEqual(r('if 1 then true else false'), data.TRUE)
        self.assertEqual(r('if 1 then true else false'), data.TRUE)
        self.assertEqual(r('if 0 then true else false'), data.TRUE)

    def test_structures(self):
        r = self.readEvalPrint
        self.assertEqual(r('()'), [])
        self.assertEqual(r('(1, 2)'), [1, 2])
        self.assertEqual(r('(1, 2, 3)'), [1, 2, 3])

        xs = [
            '1 :: []',
            '[1 2 3]',
            '[1 . [2 . [3 . nil]]]',
            '[1 . 2]',
            '[[1 2] [3 4]]',
            '[(1, 2) (3, 4)]',
            '([1 2], [3 4])',
            '[(1 + 2) (3 - 4)]',
            '(1 + 2, 3 - 4)'
        ]

        for x in xs:
            self.readEvalPrint(x)

    def test_minmax(self):
        r = self.readEvalPrint
        self.assertEqual(r('min(3, 1, 2)'), 1)
        self.assertEqual(r('max(3, 1, 2)'), 3)

    def test_comments(self):
        xs = [
            '# Line comment',
            '#| Block comment |#',
            '#| Nested #| Block comment |# |#',
            '#| Nested #| Block |# comment |#',
            '#| Line comment inside # Block comment |#',
            '# Block comment inside #| Line comment |#',
            '#',
            '#||#'
        ]

        for x in xs:
            self.assertIsNone(self.readEvalPrint(x))

    def test_string(self):
        r = self.readEvalPrint
        self.assertEqual(r('""'), '')
        self.assertEqual(r('"String"'), 'String')
        self.assertEqual(r('""""'), '"')

    def test_print(self):
        xs = [
            'print 1',
            'print "a"',
            'let x := 1 and y := 2 in print (x, y)'
        ]

        for x in xs:
            self.readEvalPrint(x)

if __name__ == '__main__':
    unittest.main()
