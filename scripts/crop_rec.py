#   CROP RECOMMENDATION MODEL
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pickle

#   Data set loading
file_path = "C:\\DataSetFolder\\Crop_recommendation.csv"
crop_data = pd.read_csv(file_path)

#   Dataset exploration
print("First 5 rows of the dataset:\n", crop_data.head())
print("\nMissing_values in the dataset:\n", crop_data.isnull().sum())
print("\nSummary statistics of the dataset:\n", crop_data.describe())

#   Target labels encoding
label_encoder = LabelEncoder()
crop_data['label_encoded'] = label_encoder.fit_transform(crop_data['label'])

#   Mapping btw encodaded labels and crop names
label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
print("Label Mapping:\n", label_mapping)

#   Seperate features and target variables
X = crop_data.drop(columns=['label', 'label_encoded'])
y = crop_data['label_encoded']

# Log training data column order
print("Training data columns:", X.columns.to_list())

#   Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#   Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

#   I am using a random forest model

#   Train a random forest classifier
random_forest = RandomForestClassifier(random_state=42, n_estimators=100)
random_forest.fit(X_train, y_train)

#   Predict on the test set
y_pred = random_forest.predict(X_test)

#   Decode predictions to crop name for interpretation
decoded_predictions = [label_mapping[pred] for pred in y_pred]

#   Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
class_rept = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
conf_matx = confusion_matrix(y_test, y_pred)

#   View the results
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:\n", class_rept)
print("confusion Matrix:\n", conf_matx)

#   Feature importance
feature_importance = random_forest.feature_importances_

#   Visualize with a bar chat
features = X.columns
indices = np.argsort(feature_importance)[::-1]

# Plot
plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.barh(features[indices], feature_importance[indices], align="center")
plt.xlabel("Relative Importance")
plt.ylabel("Features")
plt.gca().invert_yaxis()    #Reverse the order for beter visualization
plt.show()

# #   Decode prediction to crop name
# predicted_crop = label_mapping[new_pred[0]]
# print(f"The predicted crop for the given condition is: {predicted_crop}")

#   Save the model and associated files
with open('crop_recommendation_model.pkl', 'wb') as model_file:
    pickle.dump(random_forest, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

with open('label_encoder.pkl', 'wb') as encoder_file:
    pickle.dump(label_encoder, encoder_file)

#   Load the model and associated files
with open('crop_recommendation_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

with open('label_encoder.pkl', 'rb') as encoder_file:
    loaded_encoder = pickle.load(encoder_file)

# Define the expected feature order as per the training data
expected_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph','rainfall']

#   Function to get user input for prediction
def get_user_input():
    try:
        print("\nEnter the following values for prediction: ")
        n = float(input("Nitrogen (N): "))
        p = float(input("Phosphorous (P): "))
        k = float(input("Potassiun (K): "))
        temperature = float(input("Temperature (°C): "))
        humidity = float(input("Relative Humidity (%): "))
        ph = float(input("Soil pH: "))
        rainfall = float(input("Rainfall (mm): "))

        #   Create a DataFrame with proper feature names
        data = pd.DataFrame({
            'N': [n],
            'P': [p],
            'K': [k],
            'temperature': [temperature],
            'humidity': [humidity],
            'ph': [ph],
            'rainfall': [rainfall]
        }, columns=expected_features)
        return data
    except ValueError:
        print("Invalid inputs. Please enter numeric values only")
        return None

#   Function to predict crop based on user input
def predict_crop(input_features, model, scaler, label_encoder):
    if input_features is not None:
        #   Scale the features
        scaled_features = scaler.transform(input_features)
        #   Predict using the model
        prediction = model.predict(scaled_features)
        #   Decode predictions
        crop_name = label_encoder.inverse_transform(prediction)
        return crop_name[0]   
    else:
        return None 

if __name__ == "__main__":
    # Prompt the user for input
    input_features = get_user_input()

    if input_features is not None:
        # Make a prediction
        predicted_crop = predict_crop(input_features, random_forest, scaler, label_encoder)

        # Display the result
        print(f"The recommended crop for the given conditions is: {predicted_crop}")
    else:
        print("Failed to provide valid input.")
