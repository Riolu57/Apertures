from conceptor_test_class import ConceptorTestBase
import numpy as np


class TestConceptor(ConceptorTestBase):
    def test_projection(self) -> None:
        # If the conceptor works, driving the resorvoir with the signal, filtering with a conceptor without a
        # loaded connection matrix should work
        res = list()

        # For each signal
        for sig_idx in range(len(self.C.sig)):
            # Reset x(n)
            self.x = np.zeros((self.C.n, 1))

            # Reset X'
            x_prime = self.x

            # For each sample in the signal
            for idx in range(len(self.C.sig[sig_idx])):
                # Drive the RNN with conceptor and unloaded matrix
                self.x = self.C.conceptors[sig_idx]@np.tanh(self.C.Wstar@self.x +
                                                            self.C.Win*self.C.sig[sig_idx][idx] +
                                                            self.C.b)
                # And save the state activation in x_prime
                x_prime = np.concatenate((x_prime, self.x), axis=1)

            # Delete the washout period from x_prime
            x_prime = np.delete(x_prime, slice(self.C.washout), 1)

            # Save x_prime in the state response
            res.append(x_prime)

        for i in range(len(res)):
            resp = np.delete(self.C.state_response[i], slice(self.C.washout), 1)
            self.assertEqual(res[i].shape, resp.shape)
            self.assertTrue(self.errors(res[i], resp, acc=2))

    def test_shape(self) -> None:
        # Assert that the Conceptor is of shape
        for conceptor in self.C.conceptors:
            self.assertEqual(conceptor.shape, (self.C.n, self.C.n))

    def test_generation1(self) -> None:
        # Generate conceptor and compare
        rs = self.C.create_r_i()
        for idx in range(len(rs)):
            diff_conc = rs[idx]@np.linalg.inv(rs[idx] + (self.C.alpha**-2)*np.identity(self.C.n))
            self.assertTrue(self.errors(self.C.conceptors[idx], diff_conc, acc=5))

    def test_generation2(self) -> None:
        # Generate conceptor and compare
        rs = self.C.create_r_i()
        for idx in range(len(rs)):
            diff_conc = np.linalg.inv(rs[idx] + (self.C.alpha**-2)*np.identity(self.C.n))@rs[idx]
            self.assertTrue(self.errors(self.C.conceptors[idx], diff_conc, acc=5))

    def test_generation_equal(self) -> None:
        # Generate conceptor and compare
        rs = self.C.create_r_i()
        for idx in range(len(rs)):
            diff_conc_one = np.linalg.inv(rs[idx] + (self.C.alpha**-2)*np.identity(self.C.n))@rs[idx]
            diff_conc_two = rs[idx] @ np.linalg.inv(rs[idx] + (self.C.alpha**-2) * np.identity(self.C.n))
            self.assertTrue(self.errors(diff_conc_one, diff_conc_two, acc=5))

    def test_singular_values_comp_r(self) -> None:
        # Assert that the singular values are correct
        rs = self.C.create_r_i()
        for idx in range(len(rs)):
            _, s, _ = np.linalg.svd(rs[idx])
            _, t, _ = np.linalg.svd(self.C.conceptors[idx])
            self.assertTrue(self.errors(t, s/(s + self.C.alpha**-2)))

    def test_singular_values_range(self) -> None:
        # Assert singular values of C to be in range [0, 1)
        for con in self.C.conceptors:
            _, vals, _ = np.linalg.svd(con)
            for s in vals:
                self.assertTrue(1 > s >= 0)
