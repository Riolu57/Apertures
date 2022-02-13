import unittest
from Conceptor import conceptor
import numpy as np


class ConceptorTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.C = conceptor.Conceptor()
        self.C.load_patterns()
        self.C.compute_conceptors()

    def tearDown(self) -> None:
        del self.C

    @staticmethod
    def errors(array1: np.ndarray, array2: np.ndarray, acc: int = 12) -> bool:
        """
        Compares two numpy arrays up to acc-digits of accuracy.

        :param array1: The first to-be-compared numpy array.
        :param array2: The second to-be-compared numpy array.
        :param acc: The digit of significance after which all values in the arrays will be rounded.
        :return: Boolean - Whether the arrays are equal in all values if rounded to the given significance.
        """
        if len(array1.shape) == 2:
            # Reshape the arrays to be one dimensional
            arr1 = array1.reshape(1, array1.shape[0]*array1.shape[1])
            arr2 = array2.reshape(1, array2.shape[0]*array2.shape[1])

        else:
            arr1 = array1
            arr2 = array1

        # Return whether all rounded values in the arrays are equal
        return (np.round(arr1, acc) == np.round(arr2, acc)).all()