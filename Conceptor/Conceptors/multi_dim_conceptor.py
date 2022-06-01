from Conceptors.conceptor import Conceptor
import jax
import jax.numpy as np
import functools


class MultiDimConceptor(Conceptor):
    def __init__(self, signals, output, /, *, n=100, washout=100, aperture=10, plot_start=0, plot_end=100,
                 max_length=100, dim=12):

        self.output = output

        Conceptor.__init__(self, signals, n=n, washout=washout, aperture=aperture, plot_start=plot_start,
                           plot_end=plot_end, max_length=max_length, dim=dim)
        #
        # self.Win = 1.5 * (np.random.normal(0, 1, (n, self.output[0].shape[1])))


    def save_state_response(self) -> None:
        """
        Saves and computes a state response based on a saved signal.
        The washout period is needed to have the RNN adjust itself to the new signal and its dynamics.

        :return: None.
        """
        # Initialise P as a one-dimensional matrix without the washout period.
        self.P = np.asarray(self.output[0])[self.washout:]
        # Initialise the results as a list
        res = list()

        # For each other signal
        for i in range(1, len(self.output)):
            # Add the signal without the washout period to P
            self.P = np.concatenate((self.P, np.asarray(self.output[i])[self.washout:]), axis=0)

        # For each signal
        for sig in self.sig:
            # Reset x(n)
            self.x = np.zeros((self.n, 1))

            # Reset X' to x(0)
            x_prime = self.x

            # For each sample in the signal
            for idx in range(len(sig)):
                # Drive the RNN with this value
                inp = sig[idx].reshape(12, 1)
                self.drive(inp)
                # And save the state activation in x_prime
                x_prime = np.concatenate((x_prime, self.x), axis=1)

            # Save x_prime in the state response
            res.append(x_prime)

        # Save the state response in the Object
        self.state_response = res

    def drive(self, p: np.ndarray) -> None:
        """
        Computes the next state based on a driver/signal and the initial connection matrix.

        :param p: The value of the signal at the current point in time.
        :return: None.
        """
        self.x = np.tanh(self.Wstar@self.x + self.Win@p + self.b)

    @staticmethod
    def softmax(x):
        # normalized softmax
        x_norm = x - np.max(x)
        x_exp = np.exp(x_norm)
        return x_exp / np.sum(x_exp)

    def forward_bp(self, u, x_init=np.zeros(100, 1)):
        """ Loop over the time steps of the input sequence
        u:      (time, features)
        x_init: (n_res, )
        """
        x = x_init.copy()

        def apply_fun_scan(params, x, ut):
            """ Perform single step update of the network.
            x:  (n_res, )
            ut: (features, )
            """
            Win, W, Wout = params
            x = np.tanh(
                np.dot(Win, np.concatenate((np.ones(1, 1), ut))) + np.dot(W, x)
            )
            y = self.softmax(np.dot(
                Wout,
                np.concatenate((np.ones(1, 1), ut, x))
            ))
            return x, y

        f = functools.partial(apply_fun_scan, (self.Win, self.Wstar, self.Wout))
        _, Y = jax.lax.scan(f, x, u)
        return Y

    def loss(self, u, y_true):
        # cross entropy loss (see Bishop's Pattern Recognition book, page 209).
        y_pred = self.forward_bp(u)
        return -np.sum(np.sum(y_true * np.log(y_pred), axis=1)) / self.x.shape[0]

    def update(self, y_true, step_size=1e-2):
        grads = jax.grad(self.loss)((self.Win, self.W, self.Wout), self.x, y_true)
        self.Win, self.Wstar, self.Wout = [
            w - step_size * dw
            for w, dw in zip((self.Win, self.W, self.Wout), grads)
        ]
