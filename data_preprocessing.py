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


# SIMPLE ENCODING
# this method uses a simple encoding method, that assigna specific number
# to each non numeric value, such as 0 => Germany, 1 => Spain, ...
def simple_encode(matrix, column):
    lblen = LabelEncoder()

    matrix[:, column] = lblen.fit_transform(
        matrix[:, column])

    return matrix, lblen


# HOT ENCODING
# encode the columns that contain non-numeric values such as X[:, 0] and Y[:]
def hot_encode(matrix, column):
    column_trans = ColumnTransformer(
        [("Whatever", OneHotEncoder(), [column])], remainder="passthrough")
    matrix = column_trans.fit_transform(matrix)
    return matrix, column_trans


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



# ---- not preprocessor ----
# simple input handlers
# this functions force the user to only input one of two/three valid inputs
def two_option_questions(question, first_answer='Yes', second_answer='No'):
    answer = None
    first_answer_key = first_answer[0].upper()
    second_answer_key = second_answer[0].upper()
    while answer != first_answer_key and answer != second_answer_key:
        answer = input(f'{question} [{first_answer_key}]{first_answer[1:]}, [{second_answer_key}]{second_answer[1:]} ')
        answer = answer.upper()
    return answer
        
def three_option_questions(question, a, b, c):
    answer = None
    a_k = a[0].upper()
    b_k = b[0].upper()
    c_k = c[0].upper()
    
    while answer != a_k and answer != b_k and answer != c_k:
        answer = input(f'{question} [{a_k}]{a[1:]}, [{b_k}]{b[1:]}, [{c_k}]{c[1:]} ')
        answer = answer.upper()
    return answer