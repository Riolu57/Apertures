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



    # def test_conceptors(self):
    #     res_2 = list()
    #     res_3 = list()
    #
    #     for sig_idx in range(len(self.sig)):
    #         # Reset X'
    #         Y = np.zeros(1)
    #         self.x = np.zeros((self.n, 1))
    #
    #         # Reset X'
    #         X_prime = self.x
    #
    #         for idx in range(len(self.sig[sig_idx])):
    #             self.run_with_conceptor(sig_idx)
    #             X_prime = np.concatenate((X_prime, self.x), axis=1)
    #             Y = np.concatenate((Y, self.out()))
    #
    #         Y = np.delete(Y, 0)
    #         X_prime = np.delete(X_prime, slice(500), 1)
    #
    #         res_2.append(Y)
    #         res_3.append(X_prime)
    #
    #     # Conceptor output vs original signal
    #     ax = plt.subplot(2, 2, 1)
    #     ax.plot(self.space, res_2[1], label="Conceptor")
    #     ax.plot(self.space, self.sig[1], label="Signal")
    #     ax.set_title("Signal")
    #     ax.legend()
    #
    #     # Singular values enegrgy
    #     u, Sv, v = np.linalg.svd(self.conceptors[1], compute_uv=True, hermitian=True)
    #     R = self.state_response[1] @ self.state_response[1].T / self.state_response[1].shape[1]
    #     u2, Sv2, v2 = np.linalg.svd(R, compute_uv=True, hermitian=True)
    #     # print(Sv)
    #     ax1 = plt.subplot(2, 2, 2)
    #     ax1.plot(np.linspace(1, len(Sv), len(Sv)), Sv, label="Singular Values C")
    #     ax1.bar(np.linspace(1, len(Sv), len(Sv)), Sv - (Sv2 / (Sv2 + 10 ** -2)),
    #             label="Difference to predicted values")
    #     ax1.set_title("Singular values of C")
    #     ax1.legend()
    #
    #     # Driven Neuron vs original signal
    #     ax2 = plt.subplot(2, 2, 3)
    #     ax2.plot(np.linspace(-6.6933333667, 20, 1001), self.state_response[1][0][:], label="Neuron")
    #     ax2.plot(self.space, self.sig[1], label="Signal")
    #     ax2.set_title("Neuron driven by signal")
    #     ax2.legend()
    #
    #     # Plot difference in state activation over time
    #     norm_cor = np.corrcoef(self.state_response[1], res_3[1])[0, 1]
    #     ax3 = plt.subplot(2, 2, 4)
    #     ax3.plot(np.linspace(1, len(self.state_response[1][0][:]), len(self.state_response[1][0][:])), self.state_response[1][0][:],
    #              label="Normal activation")
    #     ax3.plot(np.linspace(1, len(res_3[1][0][:]), len(res_3[1][0][:])), res_3[1][0][:],
    #              label="Conceptor activation")
    #     ax3.set_title(f"State Activation Difference. cor: {norm_cor}")
    #     ax3.legend()
    #
    #     plt.show()
