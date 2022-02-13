from conceptor_test_class import ConceptorTestBase
import numpy as np


class TestConnectionMatrix(ConceptorTestBase):
    def test_loading(self) -> None:
        # TODO: If the conceptor works, driving the resorvoir with the signal, filtering with a conceptor without a
        # loaded connection matrix should work

        washout = 500
        res = list()

        # For each signal
        for sig in self.C.sig:
            # Reset x(n)
            self.C.x = np.zeros((self.C.n, 1))

            # Reset X'
            X_prime = self.C.x

            # For each sample in the signal
            for idx in range(len(sig)):
                # Drive the RNN with conceptor and unloaded matrix
                self.C.x = np.tanh(self.C.Wstar@self.C.x + self.C.Win*sig[idx] + self.C.b)
                # And save the state activation in X_prime
                X_prime = np.concatenate((X_prime, self.C.x), axis=1)

            # Delete the washout period from X_prime
            X_prime = np.delete(X_prime, slice(washout), 1)

            # Save X_prime in the state response
            res.append(X_prime)
