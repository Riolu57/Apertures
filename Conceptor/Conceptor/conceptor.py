import numpy as np
import matplotlib.pyplot as plt


class Conceptor:
    """
    This class implements a basic Conceptor Matrix based on an RNN that receives pre-defined Signals, saves them
    in an RNN's connection Matrix and then reproduces the original signals using Conceptors.

    TODO: Add variable n
    TODO: Add variable signals
    """
    def __init__(self):
        # For signal and plotting purposes
        self.space = np.linspace(-200, 200, 2001)
        # Generated signals
        self.sig = [
            np.sin(np.sqrt(2)*self.space),
            np.sin(self.space),
            np.sin(2*self.space),
            np.sin(3*self.space)
        ]

        # Amount of Neurons
        self.n = 100
        # Current state activation
        self.x = np.zeros((self.n, 1))
        # Input weights
        self.Win = 1.5*(np.random.normal(0, 1, (self.n, 1)))
        # Bias
        self.b = 0.2*(np.random.normal(0, 1, (self.n, 1)))
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

    def __compute_Wstar(self) -> np.ndarray:
        """
        Initializes the connection matrix as a spare matrix with around 10% density.
        The spectral radius of the matrix is adjusted to be 1.5

        :return: A spare matrix with density ~10% and spectral radius 1.5
        """
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
        """
        Computes the output matrix Wout, based on formula 114 of the paper.
        Its purpose is to successfully map state activations to the correct signal values.
        The formula is based on the regularized Wiener-Hopf solution, or Ridge Regression.

            Wout = ((XX' + e_out I_NxN)^-1XP')'

            X is a state activation record as a matrix:
                X = [ x_0 x_1 x_2 ... x_L]

            P is a concatenation of the signal values:
                P = [ p_0 p_1 p_2 ... p_L]

            The indices for P and X should match. Thus, the state x_1 was driven by p_0.

            The apostrophe  here is the transpose.
            e_out is a regularizer.
            I_NxN is an NxN size identity matrix, where N is the neuron count.

        :return: An output matrix that projects state activation onto a single value.
        """

        # If no activations have been saved yet, generate them
        if self.state_response is None:
            self.save_state_response()

        # Initialize X
        X = np.zeros((self.n, 1))

        # For each signal activation record
        for i in range(len(self.state_response)):
            # Add to X the record without the first state
            edited_res = np.delete(self.state_response[i], 0, 1)
            X = np.concatenate((X, edited_res), axis=1)

        # Delete the initialization
        X = np.delete(X, 0, 1)

        # Generate scaled identity matrix
        Id = 0.01*np.identity(self.n)

        H = (np.linalg.inv(X@X.T + Id)@X@self.P.T)
        return H.reshape((1, 100))

    def save_state_response(self, washout: int = 500) -> None:
        """
        Saves and computes a state response based on a saved signal.
        The washout period is needed to have the RNN adjust itself to the new signal and its dynamics.

        :param washout: The length of the washout period. Standard value is 500 steps.
        :return: None.
        """
        # Initialise P as a one-dimensional matrix without the washout period.
        self.P = np.asarray(self.sig[0])[washout:]
        # Initialise the results as a list
        res = list()

        # For each other signal
        for i in range(1, len(self.sig)):
            # Add the signal without the washout period to P
            self.P = np.concatenate((self.P, np.asarray(self.sig[i])[washout:]), axis=0)

        # For each signal
        for sig in self.sig:
            # Reset x(n)
            self.x = np.zeros((self.n, 1))

            # Reset X'
            X_prime = self.x

            # For each sample in the signal
            for idx in range(len(sig)):
                # Drive the RNN with this value
                self.drive(sig[idx])
                # And save the state activation in X_prime
                X_prime = np.concatenate((X_prime, self.x), axis=1)

            # Delete the washout period from X_prime
            X_prime = np.delete(X_prime, slice(washout), 1)

            # Save X_prime in the state response
            res.append(X_prime)

        # Save the state response in the Object
        self.state_response = res

    def load_patterns(self) -> None:
        """
        Loads the patterns given to the Object.
        This is based on formula (115) of the paper.
        Just like (114), this based on Ridge Regression, or Wiener-Hopf solution.

            W = ((X_tilde X_tilde' + e_W I_NxN)^-1 X_tilde (tanh^-1(X)-B)')'

            X is the state response:
                X = [ x_0 x_1 x_2 ... x_L]

            X_tilde is the state response shifted by 1 index:
                X_tilde = [ x_-1 x_0 x_1 x_2 x_3 ... x_L-1 ]

                x_-1 here is the state x is initialized to, which is the zero vector

            The apostrophe is interpreted as the transpose.
            e_W is regularizer.
            I_NxN is an NxN sized identity matrix where N is the amount of neurons.

            B is an NxL sized matrix which consists only of the bias vector:
                B = [ b b b ... b ]


        :return: None.
        """
        # Assert that the state response has been saved
        if self.state_response is None:
            self.save_state_response()

        # Initialise X and X_tilde
        X = np.zeros((self.n, 1))
        X_tilde = np.zeros((self.n, 1))

        # For each record
        for i in range(len(self.state_response)):
            # Create X_i by deleting the initial state x_-1
            temp2 = np.delete(self.state_response[i], 0, 1)
            # Add X_i to X
            X = np.concatenate((X, temp2), axis=1)
            # Create X_tilde_i by deleting the last state x_L
            temp = np.delete(self.state_response[i], slice(self.state_response[i].shape[1] - 1, self.state_response[i].shape[1]), 1)
            # Add X_tilde_i to X_tilde
            X_tilde = np.concatenate((X_tilde, temp), axis=1)

        # Delete the initilisation states of X_tilde and X
        X_tilde = np.delete(X_tilde, 0, 1)
        X = np.delete(X, 0, 1)

        # Create the components of the formula
        R = X_tilde@X_tilde.T
        Id = 0.0001*np.identity(self.n)
        inv = np.linalg.inv(R + Id)
        B = np.tile(self.b, (1, X_tilde.shape[1]))
        rhs = np.arctanh(X) - B

        # Overwrite W
        self.W = (inv@X_tilde@rhs.T).T

    def compute_conceptors(self, alpha: float = 10) -> None:
        """
        Computes conceptors for all given signals using formula XX ...

        :param alpha: The aperture parameter.
        :return: None.
        """
        if self.W is None:
            self.load_patterns()

        for states in self.state_response:
            R = (states@(states.T))/states.shape[1]
            if R.shape[0] != R.shape[1]: raise ValueError
            Id = (alpha**-2)*np.identity(self.n)
            C = R@np.linalg.inv(R + Id)
            C_inv = np.linalg.inv(R + Id)@R
            # assert self.test_errors(C, C_inv, 10), "Whoopsie"
            self.conceptors.append(C)
            C_rev = ((alpha**-2)*C)@np.linalg.inv(Id - C)
            # assert self.test_errors(R, C_rev, 6), "Conceptor is not correctly calculated"

    def drive(self, p: (int | float)) -> None:
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


if __name__ == "__main__":
    C = Conceptor()
    C.load_patterns()
    C.compute_conceptors()
    C.test_conceptors()
