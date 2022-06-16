import jax.numpy as np
import pandas as pd
import itertools as iter
import warnings
import pickle
import random

warnings.filterwarnings("ignore")
def new_dataframe():
    colNames = []
    for i in range(1, 13):
        colNames.append(f"coeff{i}")

    colNames.append("speaker_count")
    # colNames.append("utter_count")
    # colNames.append("frame_count")

    df = pd.DataFrame(columns=colNames)
    return colNames, df

colNames, df = new_dataframe()
print("Created Dataframe")

WINDOW_SIZE = 5

bounds = [i for i in iter.accumulate([30, 30, 30, 30, 30, 30, 30, 30, 30])]
print(bounds)

speak_count = 0
utter_count = 0
frame_count = -1

train_file = open("./data/ae.train", "r")
print("Loaded data")

for line in train_file.readlines():
    if line == "\n":
        utter_count +=1
        frame_count = -1
        speak_count = (utter_count//30)

    else:
        frame_count += 1

        row = line[:-2].split(" ")
        row.append(speak_count)
        row = np.asarray(row, dtype='float32').reshape(1, 13)
        # row.append(utter_count)
        # row.append(frame_count)

        #print(row)

        utter_idx = utter_count - bounds[speak_count - 1] if speak_count > 0 else utter_count
        # print(utter_idx)
        index = pd.MultiIndex.from_tuples([(int(speak_count), int(utter_idx), int(frame_count))], names=["speak", "utter", "frame"])
        row = pd.DataFrame(row, index=index, columns=colNames)

        #print(row)
        df = pd.concat([df, row])

print("Loaded dataframe")
train_file.close()

print(df.shape)
print(df)
ran = [31, 35, 88, 44, 29, 24, 40, 50, 29]
bounds = [i for i in iter.accumulate([31, 35, 88, 44, 29, 24, 40, 50, 29])]
print(bounds)

res = {}

speak_count = 0
utter_count = 0
frame_count = -1

test_file = open("./data/ae.test", "r")
print("Loaded test data")

for line in test_file.readlines():
    if line == "\n":
        utter_count +=1
        frame_count = -1

        if utter_count in bounds:
            speak_count = bounds.index(utter_count) + 1

        res[speak_count] = res.get(speak_count, 0) + 1

    else:
        frame_count += 1

        row = line[:-2].split(" ")
        row.append(speak_count)
        row = np.asarray(row, dtype='float32').reshape(1, 13)
        # row.append(utter_count)
        # row.append(frame_count)

        utter_idx = utter_count - bounds[speak_count - 1] if speak_count > 0 else utter_count
        # print(utter_idx)
        index = pd.MultiIndex.from_tuples([(int(speak_count), int(30 + utter_idx), int(frame_count))], names=["speak", "utter", "frame"])
        row = pd.DataFrame(row, index=index, columns=colNames)

        #print(row)
        df = pd.concat([df, row])

print("Loaded dataframe")
test_file.close()
for idx in range(len(ran)):
    print(f"{idx}: {ran[idx]}, {res[idx]}")
# Test that all occurences were counted correctly
speakers = dict()
for index, row in df.iterrows():
    speakers[index[0]] = speakers.get(index[0], 0) + 1

type(df)
df.dtypes
for i in range(1, 13):
    df = df.astype({f'coeff{i}': 'float32'})

df = df.astype({'speaker_count': 'int32'})
df.dtypes
print(res)
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df[~df.index.duplicated(keep='first')]
df.shape
colNames, windowed_df = new_dataframe()
past_utter = -1
past_frame = -1
count = 0

for index, row in df.iterrows():
    cur_utter = index[1]
    cur_frame = index[2]

    if cur_utter != past_utter:
        try:
            _, temp_df = new_dataframe()

            for i in range(WINDOW_SIZE):
                index = pd.MultiIndex.from_tuples([(int(count), int(i))])
                cur_row = pd.DataFrame(row.values.reshape(1, 13), index=index, columns=colNames)
                temp_df = pd.concat([temp_df, cur_row])

            windowed_df = pd.concat([windowed_df, temp_df])
            count += 1

        except KeyError:
            pass

    past_utter = cur_utter
    past_frame = cur_frame
speakers = dict()
for index, row in windowed_df.iterrows():
    speakers[index[0]] = speakers.get(index[0], 0) + 1

type(windowed_df)
print(windowed_df.dtypes)
for i in range(1, 13):
    windowed_df = windowed_df.astype({f'coeff{i}': 'float32'})

windowed_df = windowed_df.astype({'speaker_count': 'int32'})
print(windowed_df.dtypes)
print(res)
def splitData(dataset, split_fact):
    df1 = dataset.iloc[:int(split_fact[0]*len(dataset)),:]
    df2 = dataset.iloc[int(split_fact[0]*len(dataset)):,:]
    df3 = df2.iloc[int((split_fact[1]/(split_fact[1] + split_fact[2]))*len(df2)):,:]
    df2 = df2.iloc[:int((split_fact[1]/(split_fact[1] + split_fact[2]))*len(df2)),:]

    # t1 = df1.pop('speaker_count')
    # t2 = df2.pop('speaker_count')
    # t3 = df3.pop('speaker_count')

    # df1 = np.expand_dims(df1, 1)
    # df2 = np.expand_dims(df2, 1)
    # df3 = np.expand_dims(df3, 1)

    # return df1, t1, df2, t2, df3, t3
    return df1, df2, df3
# Convert df to DataSet
def splitWindowData(df, split):
    _, train_df = new_dataframe()
    _, test_df = new_dataframe()

    train_idx = random.sample(range(0, 640), int(640*split[0]))

    for idx in train_idx:
        for i in range(WINDOW_SIZE):
            train_df = pd.concat([train_df, df.loc[[(idx, i)]]])
            df.drop([(idx, i)], inplace=True)

    test_idx = random.sample(range(0, 640 - int(640*split[0])), int((640 - int(640*split[0]))*split[1]))
    idx_list = []
    for idx in test_idx:
        for i in range(WINDOW_SIZE):
            test_df = pd.concat([test_df, df.iloc[[idx*WINDOW_SIZE + i]]])
            idx_list.append(df.iloc[[idx*WINDOW_SIZE + i]].index[0])

    for idx in idx_list:
        df.drop(idx, inplace=True)

    return train_df, test_df, df
train_x, test_x, val_x = splitWindowData(windowed_df, [0.7, 0.2, 0.1])
replaced_df = pd.concat([train_x, test_x, val_x])
replaced_df = replaced_df[~replaced_df.index.duplicated(keep='first')]

print(len(train_x), len(test_x), len(val_x))
print((len(train_x) + len(test_x) + len(val_x)), 3200, len(replaced_df))
with open('Pickles/train_x.txt', 'wb') as tx:
    pickle.dump(train_x, tx)

# with open('Pickles/train_y.txt', 'wb') as ty:
#     pickle.dump(train_y, ty)

with open('Pickles/test_x.txt', 'wb') as tx:
    pickle.dump(test_x, tx)
#
# with open('Pickles/test_y.txt', 'wb') as ty:
#     pickle.dump(test_y, ty)

with open('Pickles/val_x.txt', 'wb') as vx:
    pickle.dump(val_x, vx)
#
# with open('Pickles/val_y.txt', 'wb') as vy:
#     pickle.dump(val_y, vy)
print("Done!")