from conceptor_test_class import ConceptorTestBase
import numpy as np


class TestRecords(ConceptorTestBase):
    def test_x_i_shape(self):
        # Assert X_i's shape is Nx(L - washout)
        for x in self.C.create_x_i():
            self.assertEqual(x.shape, (self.C.n, self.C.L - self.C.washout))

    def test_x_tilde_i_shape(self):
        # Assert X_tilde_i's shape is Nx(L - washout)
        for x in self.C.create_x_tilde_i():
            self.assertEqual(x.shape, (self.C.n, self.C.L - self.C.washout))

    def test_shape_states(self) -> None:
        # Assert that the state response is Nx(L + 1) [+1 for the initial 0 state]
        for response in self.C.state_response:
            self.assertEqual(response.shape, (self.C.n, self.C.L + 1))

    def test_reverse_generate_r1(self) -> None:
        # Re-generate R from conceptors as alpha**-2 (I_NxN - C)^-1 C
        rs = self.C.create_r_i()
        for idx in range(len(self.C.conceptors)):
            inverted = np.linalg.inv(np.identity(self.C.n) - self.C.conceptors[idx])
            sol = (self.C.alpha**-2)*self.C.conceptors[idx]@inverted
            self.assertTrue(self.errors(rs[idx], sol, acc=5))

    def test_reverse_generate_r2(self) -> None:
        # Re-generate R from conceptors as alpha**-2 C (I_NxN - C)^-1
        rs = self.C.create_r_i()
        for idx in range(len(self.C.conceptors)):
            inverted = np.linalg.inv(np.identity(self.C.n) - self.C.conceptors[idx])
            sol = (self.C.alpha**-2)*inverted@self.C.conceptors[idx]
            self.assertTrue(self.errors(rs[idx], sol, acc=5))

    def test_reverse_equal(self) -> None:
        # Assert that the correlation matrix re-generate methods are equal
        for con in self.C.conceptors:
            inverted = np.linalg.inv(np.identity(self.C.n) - con)
            sol_one = (self.C.alpha**2)*inverted@con
            sol_two = (self.C.alpha**2)*con@inverted
            self.assertTrue(self.errors(sol_one, sol_two, acc=5))

    def test_length(self) -> None:
        # Assert one conceptor for each record
        self.assertTrue(len(self.C.conceptors) == len(self.C.create_r_i()))
