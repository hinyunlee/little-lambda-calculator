import unittest
from core.reader import Reader
from core.evaluator import Evaluator
from core.printer import Printer
from core import primitives
import time

def slurp(x):
    with open(x, encoding='utf-8') as f:
        return f.read()

class TestBase(unittest.TestCase):
    def setUp(self):
        self.reader = Reader()
        self.evaluator = Evaluator(primitives.extend(), self.reader.read(slurp('prelude')))
        self.printer = Printer()

    def readEvalPrint(self, x):
        t0 = time.time()
        print('Read:')
        print(x)
        code = self.reader.read(x)
        print('Eval:')
        result = self.evaluator.eval(code)
        print('Print:')
        self.printer.print(result)
        t1 = time.time()
        print('Time: {}s'.format(t1 - t0))
        print()
        return result