from data_preprocessing import read, hot_encode, simple_encode, two_option_questions, three_option_questions
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np


X, y = read(filename='diabetes_prediction_dataset.csv')

X, gender_encoder = simple_encode(X, column=0)
X, smoking_history_encoder = hot_encode(X, column=4)

x_scaler = StandardScaler()
X = x_scaler.fit_transform(X)
# y doesnt need scaling

# X = data_preprocessing.backward_eliminatation(X, y)  # reduces precision, so i didnt use it

# convert y values to categorical values
label_encoder = preprocessing.LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/3, random_state=0)

# Initialize and train the model (Random Forest Classifier in this example)
model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display more detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

classes = ['No Diabetes', 'Diabetes']
tick_marks = [0.5, 1.5]
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

plt.ylabel('True label')
plt.xlabel('Predicted label')

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white')

plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


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
        blood_glucose_level = float(input("Subject's Blood Glucose Level: [mg/dL] "))
        
        x_data = np.array([[gender, age, hypertension, heart_disease, smoking_history, BMI, HbA1c_level, blood_glucose_level]])
        
        # encode and scale
        x_data[:, 0] = gender_encoder.transform(
            x_data[:, 0])
        x_data = smoking_history_encoder.transform(x_data)
        
        x_data = x_scaler.transform(x_data)
        y_predicted = model.predict(x_data);
        print(f"This subject is: Possibly {'NOT ' if not y_predicted[0] else ''}DIABETIC.\n")