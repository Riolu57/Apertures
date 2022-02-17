import numpy as np


class Conceptor:
    """
    This class implements a basic Conceptor Matrix based on an RNN that receives pre-defined Signals, saves them
    in an RNN's connection Matrix and then reproduces the original signals using Conceptors.

    TODO: Add variable n
    TODO: Add variable signals
    """
    def __init__(self):
        # Length of signal and records
        self.L = 2001
        # For signal and plotting purposes
        self.space = np.linspace(-200, 200, self.L)
        # Generated signals
        self.sig = [
            np.sin(np.sqrt(2)*self.space),
            np.sin(self.space),
            np.sin(2*self.space),
            np.sin(3*self.space)
        ]

        # Amount of Neurons
        self.n = 100
        # Washout period
        self.washout = 500
        # Signals over time
        self.P = None
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
        # Alpha value
        self.alpha = 10

        # For testing purposes
        self.load_patterns()
        self.compute_conceptors()

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

            temp_rec is a state activation record as a matrix:
                temp_rec = [ x_0 x_1 x_2 ... x_L]

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

        temp_out = (np.linalg.inv(X@X.T + scaled_identity)@X@self.P.T)
        return temp_out.reshape((1, 100))

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
            # Create X_i by deleting the initial state x_-1
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

        # Create R_i
        for x in self.create_x_i():
            temp = (x@x.T)/(self.L - self.washout)
            res.append(temp)

        return res

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
        # Get x and x_tilde
        x_tilde = np.concatenate(self.create_x_tilde_i(), axis=1)
        x = np.concatenate(self.create_x_i(), axis=1)

        # Create the components of the formula
        r = x_tilde@x_tilde.T
        scaled_identity = 0.0001*np.identity(self.n)
        inv = np.linalg.inv(r + scaled_identity)
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
