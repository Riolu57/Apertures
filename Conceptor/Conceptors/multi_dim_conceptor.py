from Conceptors.conceptor import Conceptor
import jax
import jax.numpy as np
import functools
import sys


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

    def forward_bp(self, params, u, x_init=np.zeros((100, 1))):
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
            ut_shaped = ut.reshape((12, 1))
            x = np.tanh(
                # np.dot(Win, np.concatenate((np.ones((1,)), ut))) + np.dot(W, x)
                np.dot(Win, ut_shaped) + np.dot(W, x)
            )
            y = self.softmax(np.dot(
                Wout, x
                # np.concatenate((np.ones((1,)), ut, x))
            ))

            return x, y

        f = functools.partial(apply_fun_scan, params)
        _, Y = jax.lax.scan(f, x, u)
        return Y

    def loss(self, params, u, y_true):
        # cross entropy loss (see Bishop's Pattern Recognition book, page 209).
        y_pred = self.forward_bp(params, u)
        return -np.sum(np.sum(y_true * np.log(y_pred), axis=1)) / self.x.shape[0]

    def update(self, u, y_true, step_size=1e-2):
        grads = jax.grad(self.loss)((self.Win, self.Wstar, self.Wout), u, y_true)
        self.Win, self.Wstar, self.Wout = [
            w - step_size * dw
            for w, dw in zip((self.Win, self.Wstar, self.Wout), grads)
        ]

    def eval(self, data):
        y = 0
        total = 0
        for idx, row in data.iterrows():
            total += 1
            goal = row.filter(regex="speaker*").to_numpy()
            start_x = np.zeros((100, 1))

            coeffs = row.filter(regex="coeff*").to_numpy()
            summed_arr = np.asarray(coeffs, dtype='float32').reshape(1, 12)

            res = self.softmax(self.Wout@np.tanh(self.Win@summed_arr.reshape(12, 1) + self.Wstar@start_x))
            max_idx = 0
            max_res = -1
            for i in range(9):
                if res.at[i].get() > max_res:
                    max_res = res.at[i].get()
                    max_idx = i

            if int(max_idx) == int(goal):
                y += 1

        print(f"{round((y/total)*100, 3)}%")

    def grad_descent(self, inp_df, val_df, epochs, window_size):
        self.state_spaces = dict()
        inp_len = len(inp_df)/4

        for epoch in range(epochs):
            id = 0
            print(f"Epoch: \t{epoch + 1}")
            indices = dict()

            for idx, row in inp_df.iterrows():
                if indices.get(idx, False):
                    continue

                indices[idx] = True

                if idx[1] == 0:
                    print(f"Iteration: \t{id + 1}\t{inp_len}")
                    start_x = np.zeros((100, 1))
                    goal = row.filter(regex="speaker*").to_numpy()
                    y_goal = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0])
                    y_goal = y_goal.at[int(goal)].set(1)

                    coeffs = row.filter(regex="coeff*").to_numpy()
                    summed_arr = np.asarray(coeffs, dtype='float32').reshape(1, 12)

                    con_x = np.tanh(
                        np.dot(self.Win, summed_arr.reshape(12, 1)) + np.dot(self.Wstar, start_x)
                    )
                    y = self.softmax(np.dot(
                        self.Wout, con_x
                    ))
                    self.state_spaces[f"{epoch}_{id}_{0}"] = (con_x, y, goal)

                    for i in range(1, window_size):
                        new_idx = (idx[0], i)
                        indices[new_idx] = True

                        try:
                            cur_row = inp_df.loc[[new_idx]]
                        except KeyError:
                            print("Window size is wrong, or data is incomplete!")
                            raise KeyError

                        cur_goal = cur_row.filter(regex="speaker*").to_numpy()

                        if cur_goal != goal:
                            raise ValueError

                        cur_coeffs = cur_row.filter(regex="coeff*").to_numpy()
                        cur_coeffs = np.asarray(cur_coeffs, dtype='float32').reshape(1, 12)
                        summed_arr = np.concatenate([summed_arr, cur_coeffs], axis=0)

                        con_x = np.tanh(
                            np.dot(self.Win, cur_coeffs.reshape(12, 1)) + np.dot(self.Wstar, con_x)
                        )
                        y = self.softmax(np.dot(
                            self.Wout, con_x
                        ))
                        self.state_spaces[f"{epoch}_{id}_{i}"] = (con_x, y, goal)
                    self.update(summed_arr, y_goal)

                    id += 1

            self.eval(val_df)

        return self.state_spaces