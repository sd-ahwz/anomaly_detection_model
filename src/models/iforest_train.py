import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest as ifor
from sklearn.model_selection import TimeSeriesSplit
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import classification_report, roc_auc_score
from pathlib import Path
import pickle

TRAINING_PROPORTION = 0.67
WINDOW_SIZE = 5
DATA_PATH = Path("data/final")
DATA_FILE = Path("detected_atk.csv")
SAVED_MODEL = Path("iforest.sav")
SAVED_MODEL_PATH = Path("models")

df = pd.read_csv(DATA_PATH / DATA_FILE)

# feature engineering
df['time'] = pd.to_datetime(df['time'], yearfirst=True, infer_datetime_format=True)
df['hour'] = pd.DatetimeIndex(df['time']).hour
df['minute'] = pd.DatetimeIndex(df['time']).minute
df['second'] = pd.DatetimeIndex(df['time']).second

# drop those that are recommended in EDA
df.drop(['pump_start_output', 'tank-b_height', \
        'valve-b_open', 'writable_ops_mode_scada_input', 'anomalous_seconds'], \
        inplace=True, axis=1)

# split into training and validation set 
# use TimeSeriesSplit due to time series data
time = df.pop('time')
# split_index = int(TRAINING_PROPORTION*df.shape[0]) 
y = df.pop('is_anomalous')
# df_train, y_train = df.iloc[:split_index], y[:split_index]
# df_valid, y_valid = df.iloc[split_index:], y[split_index:]
tscv = TimeSeriesSplit(n_splits=5, gap=0)
all_splits = list(tscv.split(df))
y_pred_full = np.empty(shape=df.shape[0])
# loop through each train_index, test_index
# for train_index, test_index in all_splits:
ifor = ifor(contamination=0.2, bootstrap=False, random_state=42)

# comment following 2 lines out when starting the loop
train_index = all_splits[0][0]
test_index = all_splits[0][1] 
####################################################### 
ifor.fit(df.iloc[train_index])
pickle.dump(ifor, open(SAVED_MODEL_PATH/SAVED_MODEL, 'wb'))
# loaded_ifor = pickle.load(open(SAVED_MODEL_PATH/SAVED_MODEL, 'rb'))
# y_pred = ifor.predict(df.iloc[test_index])
# y_pred_load = loaded_ifor.predict(df.iloc[test_index])
# print("\nAsserting\n")
# assert y_pred.all() == y_pred_load.all()
# print("\nAsserted\n")

# # replace the -1, 1 with 1, 0
# y_pred[y_pred == 1] = 0
# y_pred[y_pred == -1] = 1
# y_pred_sum = np.sum(sliding_window_view(y_pred, window_shape=WINDOW_SIZE), axis = 1)
# y_pred_sum_prepend = np.concatenate((np.zeros(WINDOW_SIZE - 1), y_pred_sum))
# print("Asserting")
# assert y_pred_sum.all() == y_pred_sum_prepend[5:].all()
# print("\nAsserted\n")

# print(len(test_index))
# print(test_index[0])
# print(y_pred_sum.shape)