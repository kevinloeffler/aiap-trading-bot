import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import keras

train_test_split = 0.8
plt.rcParams["figure.figsize"] = (10, 6)  # make matplotlib figures bigger

# Import Data
dataset_raw = pd.read_csv('data/training_data.csv')
