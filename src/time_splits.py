import numpy as np
from sklearn.model_selection import train_test_split

def time_aware_splits(X, y):
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test