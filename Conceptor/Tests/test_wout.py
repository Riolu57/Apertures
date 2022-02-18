from conceptor_test_class import ConceptorTestBase
import numpy as np


class TestWout(ConceptorTestBase):
    # TODO: Test for shapes of resulting matricies

    def test_reproduction(self):
        # To save the signal
        reg_sig = np.zeros(1)

        # For each signal
        for response in self.C.state_response:
            # Initialize a temporary variable
            temp_reg_sig = np.zeros(1)

            # For each recorded state
            for j in range(response.shape[1]):
                # The output is assigned to res
                res = self.C.Wout@response[:, j]
                # And recorded in temp_reg_sig
                temp_reg_sig = np.concatenate((temp_reg_sig, res), axis=0)

            temp_reg_sig = np.delete(temp_reg_sig, slice(self.C.washout + 2), 0)
            reg_sig = np.concatenate((reg_sig, temp_reg_sig), axis=0)

        self.assertTrue(self.errors(reg_sig[1:], self.C.P, acc=1))

    def test_Wout_shape(self) -> None:
        # Assert the shape to be Nx1
        self.assertEqual(self.C.Wout.shape, (1, self.C.n))
