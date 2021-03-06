import jax.numpy as np
import matplotlib.pyplot as plt
import jax.random as random
import random as py_random


class Conceptor:
    """
    This class implements a basic Conceptors Matrix based on an RNN that receives pre-defined Signals, saves them
    in an RNN's connection Matrix and then reproduces the original signals using Conceptors.

    TODO: Add variable n
    TODO: Add variable signals
    """
    __slots__ = ("key", "L", "start", "end", "space", "sig", "n", "washout", "P", "x", "Win", "Wout", "Wstar", "W", "b",
                 "state_response", "conceptors", "alpha", "dim")

    def __init__(self, signals, /, *, n=100, washout=100, aperture=10, plot_start=0, plot_end=100, max_length=100,
                 dim=1):
        # For jax random stuff
        self.key = random.PRNGKey(42)
        # Length of signal and records
        self.L = max_length
        # For signal and plotting purposes
        self.start = plot_start
        self.end = plot_end
        self.space = np.linspace(self.start, self.end, self.L)
        # Generated signals
        self.sig = signals
        # Amount of Neurons
        self.n = n
        # Output dimensions
        self.dim = dim
        # Washout period
        self.washout = washout
        # Signals over time
        self.P = None
        # Current state activation
        self.x = np.zeros((self.n, 1))
        # Input weights
        self.Win = 1.5*(random.normal(self.key, (self.n, dim)))
        # Bias
        self.b = 0.2*(random.normal(self.key, (self.n, 1)))
        # Initial W (aka W*)
        self.Wstar = self.__compute_Wstar()
        # Actual W
        self.W = None
        # To save the state response of the signals
        self.state_response = None
        # Output weights
        self.Wout = self.__compute_Wout()
        # The conceptors for the signals
        self.conceptors = list()
        # Alpha value
        self.alpha = aperture

        # # For testing purposes
        # self.load_patterns()
        # self.compute_conceptors()

    def __compute_Wstar(self) -> np.ndarray:
        """
        Initializes the connection matrix as a spare matrix with around 10% density.
        The spectral radius of the matrix is adjusted to be 1.5

        :return: A spare matrix with density ~10% and spectral radius 1.5
        """
        # Initialize connection matrix as 0 matrix
        w = np.zeros((self.n, self.n))

        # For 10% of the entries replace with a value from a N(0,1) distribution
        for _ in range(int(0.1*(self.n**2))):
            cord_1 = py_random.randrange(self.n)
            cord_2 = py_random.randrange(self.n)
            w = w.at[(cord_1, cord_2)].set(py_random.gauss(0, 1))

        # Ensure 1.5 spectral radius
        w = 1.5*w/max(abs(np.linalg.eigvals(w)))

        return w

    def __compute_Wout(self) -> np.ndarray:
        """
        Computes the output matrix Wout, based on formula 114 of the paper.
        Its purpose is to successfully map state activations to the correct signal values.
        The formula is based on the regularized Wiener-Hopf solution, or Ridge Regression.

            Wout = ((XX' + e_out I_NxN)^-1XP')'

            X is a state activation record as a matrix:
                X = [ x_0 x_1 x_2 ... x_L]

            P is a concatenation of the signal values:
                P = [ p_0 p_1 p_2 ... p_L]

            The indices for P and temp_rec should match. Thus, the state x_1 was driven by p_0.

            The apostrophe  here is the transpose.
            e_out is a regularizer.
            I_NxN is an NxN size identity matrix, where N is the neuron count.

        :return: An output matrix that projects state activation onto a single value.
        """

        # Generate X
        X = np.concatenate(self.create_x_i(), axis=1)

        # Generate scaled identity matrix
        scaled_identity = 0.01*np.identity(self.n)

        if self.dim == 1:
            temp_out = (np.linalg.inv(X@X.T + scaled_identity)@X@self.P.T)
            return temp_out.reshape((self.dim, self.n))

        else:
            temp_out = (np.linalg.inv(X@X.T + scaled_identity)@X@self.P).T
            return temp_out


    def save_state_response(self) -> None:
        """
        Saves and computes a state response based on a saved signal.
        The washout period is needed to have the RNN adjust itself to the new signal and its dynamics.

        :return: None.
        """
        # Initialise P as a one-dimensional matrix without the washout period.
        self.P = np.asarray(self.sig[0])[self.washout:]
        # Initialise the results as a list
        res = list()

        # For each other signal
        for i in range(1, len(self.sig)):
            # Add the signal without the washout period to P
            self.P = np.concatenate((self.P, np.asarray(self.sig[i])[self.washout:]), axis=0)

        # For each signal
        for sig in self.sig:
            # Reset x(n)
            self.x = np.zeros((self.n, 1))

            # Reset X' to x(0)
            x_prime = self.x

            # For each sample in the signal
            for idx in range(len(sig)):
                # Drive the RNN with this value
                self.drive(sig[idx])
                # And save the state activation in x_prime
                x_prime = np.concatenate((x_prime, self.x), axis=1)

            # Save x_prime in the state response
            res.append(x_prime)

        # Save the state response in the Object
        self.state_response = res

    def create_x_i(self) -> list:
        """
        Computes all X_i and returns a list containing all of them.

        :return: List containing all X_i
        """
        # Assert that the state response has been saved
        if self.state_response is None:
            self.save_state_response()

        res = list()

        # For each record
        for response in self.state_response:
            # Create X_i by deleting the initial state x_-1 and the washout
            x_i = np.delete(response, slice(self.washout + 1), 1)
            # Add X_i to the result
            res.append(x_i)

        return res

    def create_x_tilde_i(self) -> list:
        """
        Computes all X_tilde_i and returns a list containing all of them.

        :return: List containing all X_tilde_i
        """
        # Assert that the state response has been saved
        if self.state_response is None:
            self.save_state_response()

        res = list()

        # For each record
        for response in self.state_response:
            # Delete washout period
            temp = np.delete(response, slice(self.washout), 1)
            # Create X_tilde_i by deleting the last state x_L
            temp = np.delete(temp, slice(temp.shape[1] - 1, temp.shape[1]), 1)
            # Add X_tilde_i to res
            res.append(temp)

        return res

    def create_r_i(self) -> list:
        """
        Computes all correlations matricies R.

        :return: List containing all correlation matricies.
        """
        # For results
        res = list()

        # To not be recomputed
        Xs = self.create_x_i()

        # Create R_i
        for idx in range(len(Xs)):
            temp = (Xs[idx]@Xs[idx].T)/(self.sig[idx].shape[0] - self.washout)
            res.append(temp)

        return res

    def reg_sig_with_conc(self, idx: int):
        """
        Runs the network with a conceptor and loaded matrix for the duration of the signal.
        Tries to regenerate the original signal whose index was given.

        :param idx: The index of the signal that should be regenerated.
        :return: Array with the regenerated signal saved.
        """

        if self.conceptors == list():
            self.compute_conceptors()

        # Reset x
        self.x = np.zeros((self.n, 1))

        # For storage
        res = np.zeros((1, 1))

        # For each signal value
        for _ in self.sig[idx]:
            # Step once
            self.run_with_conceptor(idx)
            # And add the result to res
            res = np.concatenate((res, self.out()), axis=0)

        return np.delete(res, 0, 0)

    def reg_sig_with_conc_not_loaded(self, idx):
        """
        Regenerates the state response with a conceptor and returns a concatenated matrix with all values.

        :param idx: The index of the response to recreate.
        :return: A matrix containing the regenerated states.
        """
        if self.conceptors is None:
            self.compute_conceptors()

        x = np.zeros((self.n, 1))
        rec = np.zeros((1, 1))

        for val in self.sig[idx]:
            # Step once
            x = self.conceptors[idx]@np.tanh(self.Wstar@x + self.Win*val + self.b)
            # Rec new state
            rec = np.concatenate((rec, self.Wout@x), axis=0)

        return np.delete(rec, 0, 0)

    def load_patterns(self) -> None:
        """
        Loads the patterns given to the Object.
        This is based on formula (115) of the paper.
        Just like (114), this based on Ridge Regression, or Wiener-Hopf solution.

            W = ((x_tilde x_tilde' + e_W I_NxN)^-1 x_tilde (tanh^-1(x)-bias_matrix)')'

            x is the state response:
                x = [ x_0 x_1 x_2 ... x_L]

            x_tilde is the state response shifted by 1 index:
                x_tilde = [ x_-1 x_0 x_1 x_2 x_3 ... x_L-1 ]

                x_-1 here is the state x is initialized to, which is the zero vector

            The apostrophe is interpreted as the transpose.
            e_W is regularizer.
            I_NxN is an NxN sized identity matrix where N is the amount of neurons.

            bias_matrix is an NxL sized matrix which consists only of the bias vector:
                bias_matrix = [ b b b ... b ]


        :return: None.
        """

        # Get x, x_tilde and r
        x_tilde = np.concatenate(self.create_x_tilde_i(), axis=1)
        x = np.concatenate(self.create_x_i(), axis=1)

        # Create the components of the formula
        x_sq = x_tilde@x_tilde.T
        scaled_identity = 0.001*np.identity(self.n)
        inv = np.linalg.inv(x_sq + scaled_identity)
        bias_matrix = np.tile(self.b, (1, x_tilde.shape[1]))
        rhs = np.arctanh(x) - bias_matrix

        # Overwrite W
        self.W = (inv@x_tilde@rhs.T).T

    def compute_conceptors(self) -> None:
        """
        Computes conceptors for all given signals using formula XX ...

        :return: None.
        """
        if self.W is None:
            self.load_patterns()

        for r in self.create_r_i():
            scaled_id = (self.alpha**-2)*np.identity(self.n)
            conceptor = r@np.linalg.inv(r + scaled_id)
            self.conceptors.append(conceptor)

    @staticmethod
    def nrmse(p: np.ndarray, o: np.ndarray) -> float:
        """
        Computes the normalised root mean squared error.

        :param p: The predicted signal.
        :param o: The observed signal.
        :return: The NRMSE between p and o.
        """
        if p.shape != o.shape:
            raise ValueError

        return np.sqrt(sum((p - o)**2)/max(p.shape))/(max(o) - min(o))

    def shift_max(self, org: np.ndarray, imi: np.ndarray, freq: float = 1) -> np.ndarray:
        """
        Shifts imi such that the maxima of org and imi are at the same index.
        Supposed to eliminate phase shifting.

        :param org: The original signal.
        :param imi: The imitated signal.
        :param freq: The frequency of the signal.
        :return: A shifted numpy array.
        """
        sample_freq = (self.L - 1)/(self.end - self.start)
        reg = int(sample_freq/freq*2*np.pi) + 1

        # Find max in first region
        fm = np.where(org[:reg] == max(org[:reg]))[0]
        sm = np.where(imi[:reg] == max(imi[:reg]))[0]

        ph_dif = np.abs(fm - sm)[0]
        return np.delete(imi, slice(ph_dif))

    def regenerate_signals(self):
        """
        Regenerates all signals using W_out and recorded state matricies.

        :return: List containing all signals to the original signal.
        """
        if self.state_response is None:
            self.save_state_response()

        # To save the signals
        reg_sig = list()

        # For each signal
        for response in self.state_response:
            # Initialize a temporary variable
            temp_reg_sig = np.zeros(1)

            # For each recorded state
            for j in range(response.shape[1]):
                # Record in temp_reg_sig
                temp_reg_sig = np.concatenate((temp_reg_sig, self.Wout@response[:, j]), axis=0)

            # Delete two 0 states; One from temp_reg_sig init and one from the response
            temp_reg_sig = np.delete(temp_reg_sig, slice(0, 2), 0)
            reg_sig.append(temp_reg_sig)

        return reg_sig

    def drive(self, p: float) -> None:
        """
        Computes the next state based on a driver/signal and the initial connection matrix.

        :param p: The value of the signal at the current point in time.
        :return: None.
        """
        self.x = np.tanh(self.Wstar@self.x + self.Win*p + self.b)

    def run(self) -> None:
        """
        Runs the RNN using its loaded connection Matrix.

        :return: None.
        """
        self.x = np.tanh(self.W@self.x + self.b)

    def run_with_conceptor(self, idx: int) -> None:
        """
        Computes the next internal state using a loaded connection matrix and a computed conceptor.

        :param idx: The index of the signal we want to replicate/of the conceptor we want to use.
        :return: None.
        """
        self.x = self.conceptors[idx]@np.tanh(self.W@self.x + self.b)

    def out(self) -> np.ndarray:
        """
        Computes the output y(n) using the computed output matrix Wout and the current state activation.

        :return: A scalar which is the intended signal value at time n.
        """
        return self.Wout@self.x

    # def get_new_sig(self):
    #     if not self.conceptors:
    #         self.compute_conceptors()
    #
    #
    #
    #     self.sig.append(smth)
    #
    #     """
    #     Saves and computes a state response based on a saved signal.
    #     The washout period is needed to have the RNN adjust itself to the new signal and its dynamics.
    #
    #     :return: None.
    #     """
    #     # Initialise P as a one-dimensional matrix without the washout period.
    #     self.P = np.asarray(self.sig[0])[self.washout:]
    #     # Initialise the results as a list
    #     res = list()
    #
    #     # For each other signal
    #     for i in range(1, len(self.sig)):
    #         # Add the signal without the washout period to P
    #         self.P = np.concatenate((self.P, np.asarray(self.sig[i])[self.washout:]), axis=0)
    #
    #     # For each signal
    #     for sig in self.sig:
    #         # Reset x(n)
    #         self.x = np.zeros((self.n, 1))
    #
    #         # Reset X' to x(0)
    #         x_prime = self.x
    #
    #         # For each sample in the signal
    #         for idx in range(len(sig)):
    #             # Drive the RNN with this value
    #             self.drive(sig[idx])
    #             # And save the state activation in x_prime
    #             x_prime = np.concatenate((x_prime, self.x), axis=1)
    #
    #         # Save x_prime in the state response
    #         res.append(x_prime)
    #
    #     # Save the state response in the Object
    #     self.state_response = res

    def reset(self):
        self.x = np.zeros((self.n, 1))

    def vis_conceptor(self):
        res_2 = self.regenerate_signals()
        rs = self.create_r_i()

        # Conceptors output vs original signal
        ax = plt.subplot(2, 2, 1)
        pacing = self.L
        space = self.space
        y1 = res_2[1]
        y2 = self.sig[1]
        y3 = self.reg_sig_with_conc(1)
        y4 = self.reg_sig_with_conc_not_loaded(1)
        ax.plot(space, y1, label="W_out", linestyle='dashed', color="black")
        ax.plot(space, y2, label="Signal", color="grey", alpha=0.6)
        ax.plot(space, y3, label="Conceptors", color="blue", alpha=0.5, linestyle="dashdot")
        ax.plot(space, y4, label="Unloaded Conceptors", color="green", alpha=0.5)
        nrmse1 = self.nrmse(y1.reshape(y1.shape[0], 1), y2.reshape(y2.shape[0], 1))
        nrmse2 = self.nrmse(y3.reshape(y3.shape[0], 1), y2.reshape(y2.shape[0], 1))
        nrmse3 = self.nrmse(y4.reshape(y4.shape[0], 1), y2.reshape(y2.shape[0], 1))
        ax.set_title("Signal")
        textstr = f"NRMSE out/sig: {round(float(nrmse1), 3)}\nNRMSE con/sig {round(float(nrmse2), 3)}\nNRMSE unloaded/"\
            + f"sig {round(float(nrmse3), 3)}"
        props = dict(boxstyle='round', facecolor='grey', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        ax.legend()

        del y1, y2, y3, y4, nrmse1, nrmse2, nrmse3, textstr, props

        # Singular values enegrgy
        u, Sv, v = np.linalg.svd(self.conceptors[1], compute_uv=True, hermitian=True)
        u2, Sv2, v2 = np.linalg.svd(rs[1], compute_uv=True, hermitian=True)
        # print(Sv)
        ax1 = plt.subplot(2, 2, 2)
        ax1.plot(np.linspace(1, len(Sv), len(Sv)), Sv, label="Singular Values C")
        ax1.bar(np.linspace(1, len(Sv), len(Sv)), Sv - (Sv2 / (Sv2 + 10 ** -2)),
                label="Difference to predicted values")
        ax1.set_title("Singular values of C")
        ax1.legend()

        del u, Sv, v, u2, Sv2, v2

        # Driven Neuron vs original signal
        ax2 = plt.subplot(2, 2, 3)
        temp = np.delete(self.state_response[1], 0, 1)
        ax2.plot(space, temp[0][self.L - pacing:], label="Neuron")
        ax2.plot(space, self.sig[1][self.L - pacing:], label="Signal")
        ax2.set_title("Neuron driven by signal")
        ax2.legend()

        del temp

        # Display tanh(W^*_i x^j(n) + W^in_i p^j(n + 1) + b_i) \approx tanh(W_i x^j(n) + b_i)
        ax3 = plt.subplot(2, 2, 4)
        diffs = list()
        for resp in range(len(self.state_response)):
            diff = np.zeros((1, 1))
            for idx in range(self.sig[resp].shape[0] - self.washout):
                x = self.state_response[resp][:, self.washout + idx]
                p = self.P[resp*(self.L - self.washout) + idx]
                temp = np.sum(np.tanh(self.Wstar@x + self.Win*p + self.b) - np.tanh(self.W@x + self.b)).reshape((1, 1))
                diff = np.concatenate((diff, temp), axis=0)
            diffs.append(diff.reshape((1, diff.shape[0])))

        col = ['red', 'blue', 'yellow', 'pink', 'cyan', 'orange']
        data = np.concatenate(diffs, axis=0)
        X = np.arange(data.shape[1])
        for i in range(data.shape[0]):
            ax3.bar(X, data[i], bottom=np.sum(data[:i],
                                              axis=0), color=col[i % len(col)])
        ax3.hlines(np.mean(data), 0, self.L)

        del diffs, diff, x, p, temp, col, data, X

        plt.show()

    # def vis_attack(self):
    #     ax = plt.subplot(2, 2, 1)
    #     ax.

if __name__ == "__main__":
    space = np.linspace(-200, 200, 2001)
    signals = [
            np.sin(np.sqrt(2)*space),
            np.sin(space),
            np.sin(2*space),
            np.sin(3*space)
        ]

    C = Conceptor(signals, n=100, washout=500, aperture=10, plot_start=-200, plot_end=200, max_length=2001, dim=1)

    C.load_patterns()
    C.vis_conceptor()


