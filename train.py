import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
data = pd.read_excel('Energy Efficiency Dataset.xlsx', sheet_name='sheet1', engine='openpyxl')
print("Dataset before handling missing values:")
print(data)


# Drop unnecessary columns
data = data.drop(columns=["Unnamed: 9", "Unnamed: 10", "Unnamed: 11"])

# Step 1: Check for missing values
if data.isnull().sum().any():
    print("Dataset contains missing values. Handling them by dropping rows with nulls.")
    data = data.dropna()

print("Dataset after handling missing values:")
print(data)
if data.empty:
    raise ValueError("The dataset is empty after dropping rows with missing values.")

# Step 2: Separate features and target
features = data[[
    "Relative Compactness",
    "Surface Area",
    "Wall Area",
    "Roof Area",
    "Overall Height",
    "Orientation",
    "Glazing Area",
    "Glazing Area Distribution",
]]
target = data["Heating Load"]

# Step 3: Normalize the features
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_normalized, target, test_size=0.2, random_state=42)

# Step 5: Train the AI Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the Model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")

# Step 7: Save the Model (optional)
import joblib
joblib.dump(model, 'heating.pkl')
joblib.dump(scaler, 'scaler.pkl')
# Step 8: Make Predictions (example)
def predict_heating_load(input_data):
    normalized_data = scaler.transform([input_data])
    return model.predict(normalized_data)[0]

# Example input: [0.98, 514.5, 294.0, 110.25, 7.0, 2, 0.0, 0]
example_input = [0.98, 514.5, 294.0, 110.25, 7.0, 2, 0.0, 0]
predicted_value = predict_heating_load(example_input)
print(f"Predicted Heating Load for input {example_input}: {predicted_value}")

feature_importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': features.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("Feature Importance:")
print(feature_importance_df)
# Save the cleaned and normalized dataset (optional)
prepared_data = pd.DataFrame(features_normalized, columns=features.columns)
prepared_data['Heating Load'] = target.values
prepared_data.to_csv('cleaned_energy_efficiency_dataset.csv', index=False)

print("Dataset has been cleaned, normalized, and saved as 'cleaned_energy_efficiency_dataset.csv'.")
