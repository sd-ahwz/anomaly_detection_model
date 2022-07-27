import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest as ifor
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path
import pickle

MODEL_PATH = Path("models")
MODEL_FILE = Path("iforest.sav")
WINDOW_SIZE = 5

ifor = pickle.load(open(MODEL_PATH/MODEL_FILE, 'rb'))

# ingest df of test data properly cleaned
anomaly = ifor.predict(df)
