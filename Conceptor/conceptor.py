from array import array
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt


class Conceptor():
    def __init__(self):
        # Whether state activations have been saved
        self.saved = False
        # Amount of Neurons
        self.n = 100
        # State activation
        self.x = np.zeros((self.n, 1))
        # Input weights
        self.Win = 1.5*(np.random.normal(0, 1, (self.n, 1)))
        # Bias
        self.b = 0.2*(np.random.normal(0, 1, (self.n, 1)))
        # Initial W (aka W*)
        self.Wstar = self.__compute_W()
        # For signal and plotting purposes
        self.space = np.linspace(-20, 20, 1500)

        # Generated signals
        self.sig = [
            np.sin(np.sqrt(2)*self.space),
            np.sin(self.space),
            np.sin(2*self.space),
            np.sin(3*self.space)
        ]

        # Output weights
        self.Wout = self.__compute_Wout()

        # The conceptors for the signals
        self.conceptors = list()

    def __compute_W(self) -> np.ndarray:
        # Initialize connection matrix as 0 matrix
        w = np.random.normal(0, 1, (self.n, self.n))

        # For 10% of the entries replace with a value from a N(0,1) distribution
        for _ in range(int(0.1*(self.n**2))):
            cord = np.random.choice(self.n, size=2, replace=True)
            w[(cord[0], cord[1])] = np.random.normal(0, 1)

        # Ensure 1.5 spectral radius
        w = 1.5*w/max(abs(np.linalg.eigvals(w)))

        return w

    def __compute_Wout(self) -> np.ndarray:
        # If no activations have been saved yet, generate them
        if not self.saved:
            self.save_state_response()

        # Initialize X
        X = np.zeros((self.n, 1))

        # For each activation record
        for i in range(len(self.res)):
            edited_res = np.delete(self.res[i], 0, 1)
            X = np.concatenate((X, edited_res), axis=1)

        Id = 0.01*np.identity(self.n)

        H = (np.linalg.inv(X@X.T + )@X@self.P.T).T
        return H

    def drive(self, p: (int | float)) -> None:
        self.x = np.tanh(self.Wstar@self.x + self.Win*p + self.b)

    def save_state_response(self, washout: int=500) -> None:
        self.P = np.asarray(self.sig[0])[washout:]
        res = list()
        
        for i in range(1, len(self.sig)):
            self.P = np.concatenate((self.P, np.asarray(self.sig[i])[washout:]), axis=0)
            print(self.P)

        for sig in self.sig:
            # Reset x(n)
            self.x = np.zeros((self.n, 1))

            # Reset X'
            X_prime = self.x

            for idx in range(len(sig)):
                self.drive(sig[idx])
                X_prime = np.concatenate((X_prime, self.x), axis=1)

            X_prime = np.delete(X_prime, slice(washout), 1)

            res.append(X_prime)

        self.res = res
        self.save = True

    def load_patterns(self):
        if not self.saved:
            self.save_state_response()

        X = np.zeros((self.n, 1))
        X_tilde = np.zeros((self.n, 1))

        for i in range(len(self.res)):
            temp2 = np.delete(self.res[i], 0, 1)
            X = np.concatenate((X, temp2), axis=1)
            temp = np.delete(self.res[i], slice(self.res[i].shape[1] - 1, self.res[i].shape[1]), 1)
            X_tilde = np.concatenate((X_tilde, temp), axis=1)

        X_tilde = np.delete(X_tilde, 0, 1)
        X = np.delete(X, 0, 1)

        R = X_tilde@X_tilde.T
        Id = 0.0001*np.identity(self.n)
        inv = np.linalg.inv(R + Id)
        B = np.tile(self.b, (1, X_tilde.shape[1]))
        rhs = np.arctanh(X) - B

        self.W = (inv@X_tilde@rhs.T).T        

    def run(self) -> None:
        self.x = np.tanh(self.W@self.x + self.b)

    def run_with_conceptor(self, idx: int):
        self.x = self.conceptors[idx]@np.tanh(self.W@self.x + self.b)

    def out(self) -> np.ndarray:
        return self.Wout@self.x

    def compute_conceptors(self, alpha: float=10):
        if not hasattr(self, 'W'):
            self.load_patterns()

        for states in self.res:
            R = (states@(states.T))/states.shape[1]
            if R.shape[0] != R.shape[1]: raise ValueError
            Id = (alpha**-2)*np.identity(self.n)
            C = R@np.linalg.inv(R + Id)
            C_inv = np.linalg.inv(R + Id)@R
            assert self.test_errors(C, C_inv, 10), "Whoopsie"
            self.conceptors.append(C)
            C_rev = ((alpha**-2)*C)@np.linalg.inv(Id - C)
            assert self.test_errors(R, C_rev, 6), "Conceptor is not correctly calculated"

    @staticmethod
    def test_errors(array1, array2, acc):
        arr1 = array1.reshape(1, array1.shape[0]*array1.shape[1])
        arr2 = array2.reshape(1, array2.shape[0]*array2.shape[1])

        return (np.round(arr1, acc) == np.round(arr2, acc)).all()

    def test_conceptors(self):
        res_2 = list()
        res_3 = list()

        for sig_idx in range(len(self.sig)):
            # Reset X'
            Y = np.zeros(1)
            self.x = np.zeros((self.n, 1))

            # Reset X'
            X_prime = self.x

            for idx in range(len(self.sig[sig_idx])):
                self.run_with_conceptor(sig_idx)
                X_prime = np.concatenate((X_prime, self.x), axis=1)
                Y = np.concatenate((Y, self.out()))

            Y = np.delete(Y, 0)
            X_prime = np.delete(X_prime, slice(500), 1)

            res_2.append(Y)
            res_3.append(X_prime)

        # Conceptor output vs original signal
        ax = plt.subplot(2, 2, 1)
        ax.plot(self.space, res_2[1], label="Conceptor")
        ax.plot(self.space, self.sig[1], label="Signal")
        ax.set_title("Signal")
        ax.legend()

        # Singular values enegrgy
        u, Sv, v = np.linalg.svd(self.conceptors[1], compute_uv=True, hermitian=True)
        R = self.res[1]@self.res[1].T/self.res[1].shape[1]
        u2, Sv2, v2 = np.linalg.svd(R, compute_uv=True, hermitian=True)
        # print(Sv)
        ax1 = plt.subplot(2, 2, 2)
        ax1.plot(np.linspace(1, len(Sv), len(Sv)), Sv, label="Singular Values C")
        ax1.bar(np.linspace(1, len(Sv), len(Sv)), Sv - (Sv2/(Sv2 + 10**-2)), label="Difference to predicted values")
        ax1.set_title("Singular values of C")
        ax1.legend()

        # Driven Neuron vs original signal
        ax2 = plt.subplot(2, 2, 3)
        ax2.plot(np.linspace(-6.6933333667, 20, 1001), self.res[1][0][:], label="Neuron")
        ax2.plot(self.space, self.sig[1], label="Signal")
        ax2.set_title("Neuron driven by signal")
        ax2.legend()

        # Plot difference in state activation over time
        norm_cor = np.corrcoef(self.res[1], res_3[1])[0, 1]
        ax3 = plt.subplot(2, 2, 4)
        ax3.plot(np.linspace(1, len(self.res[1][0][:]), len(self.res[1][0][:])), self.res[1][0][:], label="Normal activation")
        ax3.plot(np.linspace(1, len(res_3[1][0][:]), len(res_3[1][0][:])), res_3[1][0][:], label="Conceptor activation")
        ax3.set_title(f"State Activation Difference. cor: {norm_cor}")
        ax3.legend()


        plt.show()

        





if __name__ == "__main__":
    C = Conceptor()
    C.load_patterns()
    C.compute_conceptors()
    C.test_conceptors()