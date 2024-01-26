import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from data_preprocessing import read, hot_encode, simple_encode, two_option_questions, three_option_questions
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = read(filename='diabetes_prediction_dataset.csv', X_start=0, X_end=-2)

X, gender_encoder = simple_encode(X, column=0)
X, smoking_history_encoder = hot_encode(X, column=4)
# Split the dataset; svr doesnt autoscale and not doing this will cause calculation problems

x_scaler = StandardScaler()
X = x_scaler.fit_transform(X)
# y doesnt need scaling

# X = data_preprocessing.backward_eliminatation(X, y)  # reduces precision, so i didnt use it

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/3, random_state=0)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and train the model (Random Forest Regressor in this example)
model = RandomForestRegressor(random_state=0)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error after tuning: {mse:.2f}")
print(f"R-squared after tuning: {r2:.2f}")

# Visualize the predicted vs. actual values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Blood Glucose Level")
plt.ylabel("Predicted Blood Glucose Level")
plt.title("Actual vs. Predicted Blood Glucose Level")
plt.show()

# simple error calculator
delta = []
i = 0
for yt in y_test:
    dy = dict()
    dy['exact'] = np.abs(yt - y_pred[i])
    dy['error'] = f"{100 * dy['exact'] / y_pred[i]: .4f} %"
    delta.append(dy)
    i += 1


if __name__ == '__main__':
    print("\n\n\nNow enter the following parameters about each Subject in order to obtain single predictions:")
    while True:
        print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
        gender = two_option_questions("Gender: ", "Male", "Female")
        gender = 'Female' if gender == 'F' else 'Male'
        age = int(input('Age: '))
        
        hypertension = two_option_questions('Does Subject have Hypertension? ')
    
        hypertension = int(hypertension == 'Y') # convert to one or zero
        heart_disease = two_option_questions("Does Subject Have Had Heart Disease? ")
        heart_disease = int(heart_disease == "Y")
        smoking_history = three_option_questions("What's Subject's Smoking Status: ", "Currently a Smoker", 'Former Smoker', 'Never Smoked')
        smoking_history = "current" if smoking_history == 'C' else \
                ("former" if smoking_history == "f" else "never")
        height = float(input("Subject's Height (cm): "))
        height /= 100 # convert to meters
        weight = float(input("Subject's weight (Kg): "))
        BMI = weight / (height ** 2)
    
        HbA1c_level = float(input("Subject's HbA1c Level (average blood glucose (sugar) levels for the last two to three months):"))

        x_data = np.array([[gender, age, hypertension, heart_disease, smoking_history, BMI, HbA1c_level]])
        
        # encode and scale
        x_data[:, 0] = gender_encoder.transform(
            x_data[:, 0])
        x_data = smoking_history_encoder.transform(x_data)
        
        x_data = x_scaler.transform(x_data)
        y_predicted = model.predict(x_data);
        print(f"This subject's Blood Glucose Level is: Possibly {y_predicted[0]} mg/dL.\n")