import unittest
from test_wout import TestWout
from test_conceptor import TestConceptor


class ConceptorTests(unittest.TestSuite):
    def __init__(self):
        super().__init__()
        self.addTest(unittest.makeSuite(TestWout))
        self.addTest(unittest.makeSuite(TestConceptor))


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=1).run(ConceptorTests())
