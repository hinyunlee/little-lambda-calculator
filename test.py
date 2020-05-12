import unittest
from tests import test_basic, test_advanced

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromModule(test_basic))
    suite.addTests(loader.loadTestsFromModule(test_advanced))
    unittest.TextTestRunner().run(suite)
