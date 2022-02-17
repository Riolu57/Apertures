import unittest

import numpy as np

from conceptor_test_class import ConceptorTestBase


class TestErrorFunc(ConceptorTestBase):
    def test_true_one_d(self) -> None:
        L = [
            ([1, 2, 3], [1, 2, 3]),
            ([20, 40, 60], [20, 40, 60]),
            ([[-1], [1], [2]], [[-1], [1], [2]])
        ]
        for i in L:
            a, b = i
            x = np.asarray(a)
            y = np.asarray(b)
            self.assertTrue(self.errors(x, y))

    def test_false_one_d(self) -> None:
        L = [
            ([-1, 2, 3], [1, 2, 3]),
            ([20, -40, 60], [20, 40, 60]),
            ([[-1], [0], [2]], [[-1], [1], [2]])
        ]
        for i in L:
            a, b = i
            x = np.asarray(a)
            y = np.asarray(b)
            self.assertFalse(self.errors(x, y))

    def test_true_two_d(self) -> None:
        L = [
            (np.arange(20).reshape(5, 4), np.arange(20).reshape(5, 4)),
            (np.arange(20).reshape(5, 4), np.arange(20)),
            (np.arange(20), np.arange(20).reshape(5, 4)),
            (np.arange(20).reshape(4, 5), np.arange(20).reshape(5, 4))
            ]
        for i in L:
            a, b = i
            self.assertTrue(self.errors(a, b))

    def test_false_two_d(self) -> None:
        L = [
            (np.arange(20).reshape(5, 4), np.ones((5, 4))),
            (np.zeros((5, 4)), np.arange(20)),
            (np.ones((5, 4)), np.arange(20).reshape(5, 4))
            ]
        for i in L:
            a, b = i
            self.assertFalse(self.errors(a, b))

    def test_error_dimensions(self) -> None:
        L = [
            (np.arange(20).reshape(5, 4), np.ones((4, 4))),
            (np.arange(19), np.zeros((5, 4))),
            (np.ones((5, 4)), np.arange(16).reshape(4, 4))
            ]
        for i in L:
            a, b = i
            self.assertRaises(ValueError, self.errors, a, b)
