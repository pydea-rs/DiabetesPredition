import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import statsmodels.api as sm


# distinguish dependant and independant variables
# from the dataset structure its obvious that 'Purchased' is a dependent var (Y) and
# others are independent (X)
def read(filename: str='diabetes_prediction_dataset.csv', X_start: int=0, X_end: int = -1):
    # read the dataset
    dataset = pd.read_csv(filename)
    X = dataset.iloc[:, X_start:X_end].values
    Y = dataset.iloc[:, X_end].values
    return X, Y


def is_vector(x):
    return len(x.shape) == 1
    # return isinstance(x, (list, tuple, np.ndarray))
    # return hasattr(x, "__len__") and not np.isscalar(x[0])


def is_nan(x):
    for el in x:
        # if str(el).isnumeric() or np.isscalar(el):
        try:
            float(el)
            return False
        except:
            pass
    return True


# SIMPLE ENCODING
# this method uses a simple encoding method, that assigna specific number
# to each non numeric value, such as 0 => Germany, 1 => Spain, ...
def simple_encode(matrix, single_column=None):
    lblen = LabelEncoder()
    if single_column is not None:
        matrix[:, single_column] = lblen.fit_transform(
            matrix[:, single_column])
        return matrix

    if not is_vector(matrix):
        cols = len(matrix[0, :])
        for i in range(cols):
            if is_nan(matrix[:, i]):
                matrix[:, i] = lblen.fit_transform(matrix[:, i])
    else:
        if is_nan(matrix):
            matrix[:] = lblen.fit_transform(matrix[:])
    return matrix


# HOT ENCODING
# encode the columns that contain non-numeric values such as X[:, 0] and Y[:]
def hot_encode(matrix, single_column=None):

    def hot_encode_single(matrix, column):
        column_trans = ColumnTransformer(
            [("Whatever", OneHotEncoder(), [column])], remainder="passthrough")
        matrix = column_trans.fit_transform(matrix)
        return matrix

    if single_column is not None:
        #lblhoten = OneHotEncoder(categories_features=[single_column])
        #matrix = lblhoten.fit_transform(matrix).toarray()
        return hot_encode_single(matrix, single_column)

    if not is_vector(matrix):
        cols = len(matrix[0])
        i = 0
        while i < cols:
            if is_nan(matrix[:, i]):
                previous_length = len(matrix[0])
                matrix = hot_encode_single(matrix, i)
                cols = len(matrix[0])
                i += cols - previous_length
            i += 1

    else:
        if is_nan(matrix):
            matrix = hot_encode_single(matrix, 0)
    return matrix


def add_ones(matrix):
    return np.append(arr=np.ones((len(matrix), 1)).astype(int), values=matrix, axis=1)
    # arr and values parameters are used instead of eachother, so the ones become at the start of the matrix
  
    
def backward_eliminatation(X, y, SIGFICANT_LEVEL: float = 0.05):
    X = add_ones(X)  # REMEMBER: this doesnt change the actual X outside the function
    X_optimized = X[:, :]
    # start of Backward Elimination:
    while np.any(X_optimized):  # X_optimized is not empty
        X_optimized = np.array(X_optimized, dtype=float)  # this line fixes: ufunc 'isfinite' not supported for the input types, 
        OLS_regressor = sm.OLS(endog=y, exog=X_optimized).fit()
        print("X_opt=", X_optimized, " Summary: ", OLS_regressor.summary())
        maxp_index = np.argmax(OLS_regressor.pvalues)  # index of maximum p
        if maxp_index < SIGFICANT_LEVEL:  # end of the back-elimination algorythm
            break
        # if P > SL => remove predictor
        X_optimized = np.delete(X_optimized, maxp_index, axis=1)
    return X_optimized