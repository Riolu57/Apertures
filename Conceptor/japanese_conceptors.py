import pickle
from enum import Enum

import jax.numpy as np

from Conceptors.multi_dim_conceptor import MultiDimConceptor


class UnPickType(Enum):
    TRAIN_X = 1
    TRAIN_Y = 2
    TEST_X = 3
    TEST_Y = 4
    VAL_X = 5
    VAL_Y = 6


def unPick(type):
    f_path = "../Pickles/"
    x = type.value%2 == 1
    if type.value in [1, 2]:
        f_path += f"train_{'x' if x else 'y'}"

    elif type.value in [3, 4]:
        f_path += f"test_{'x' if x else 'y'}"

    elif type.value in [5, 6]:
        f_path += f"val_{'x' if x else 'y'}"

    else:
        raise ValueError

    f_path += ".txt"

    file = open(f_path, "rb")
    res = pickle.load(file)
    file.close()
    return res

train_x = unPick(UnPickType.TRAIN_X)
test_x = unPick(UnPickType.TEST_X)
val_x = unPick(UnPickType.VAL_X)

data = list()
out = list()
max_len = -1
min_len = 100000000
tot_len = 0

for idx in range(0, 9, 1):
    temp = train_x[train_x.speaker_count == idx].filter(regex="coeff*").to_numpy()
    temp = np.asarray(temp, dtype='float32')
    max_len = len(temp) if len(temp) > max_len else max_len
    min_len = len(temp) if len(temp) < min_len else min_len
    tot_len += len(temp)
    data.append(temp)
    x = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0])
    x = x.at[idx].set(1)
    out.append(np.tile(x, (len(temp), 1)))

print(min_len, tot_len)

print("Data!")

M = MultiDimConceptor(
    data,
    out,
    n=100,
    washout=100,
    plot_end=max_len,
    max_length=max_len
)

print("Conceptor!")

M.reset()

def lol():
    print("Conceptor reset!")
    inp = val_x[val_x.speaker_count == 0].filter(regex="coeff*").to_numpy()[0].reshape(12, 1)
    M.drive(inp)
    print("Conceptor driven!")
    res = M.out()
    print(res)
    print(np.where(res == max(res))[0][0])


M.grad_descent(train_x, val_x, 5, 5)

