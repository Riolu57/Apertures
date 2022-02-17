import unittest
from test_wout import TestWout
from test_conceptor import TestConceptor
from test_win import TestWin
from test_connection_matrix import TestConnectionMatrix
from test_correlations import TestCorrelations
from test_records import TestRecords
from test_error_function import TestErrorFunc


class ConceptorTests(unittest.TestSuite):
    def __init__(self):
        super().__init__()
        self.addTest(unittest.makeSuite(TestWout))
        self.addTest(unittest.makeSuite(TestWin))
        self.addTest(unittest.makeSuite(TestConceptor))
        self.addTest(unittest.makeSuite(TestConnectionMatrix))
        self.addTest(unittest.makeSuite(TestCorrelations))
        self.addTest(unittest.makeSuite(TestRecords))
        self.addTest(unittest.makeSuite(TestErrorFunc))


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=1).run(ConceptorTests())
