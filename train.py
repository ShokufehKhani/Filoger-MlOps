import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import re 
import warnings

warnings.filterwarnings('ignore')

# Path to save the model
MODEL_PATH = 'price_prediction_model.pkl'

# Load our dataset (assuming a CSV file or similar)

dataset = pd.read_csv("housePrice.csv")


# Change datatype of Area
dataset["Area"] = dataset["Area"].apply(lambda x : re.sub(',', '', x))
dataset["Area"] = pd.to_numeric(dataset["Area"])

# Drop missing values
dataset.dropna(inplace = True)

# Drop Duplicate values
dataset.drop_duplicates(inplace = True)
dataset.reset_index(inplace = True)
dataset.drop('index', axis = 1, inplace = True)

# Remove outliers
PriceUpperFence =  13370000000
AreaUpperFence  = 200
Price_UpperFence_Data =  np.where(dataset["Price"] > PriceUpperFence)
Area_UpperFence_Data  =  np.where(dataset["Area"] > AreaUpperFence)
Price_Area_Outliers =np.union1d(Price_UpperFence_Data , Area_UpperFence_Data)
Dataset_Without_Outliers = dataset.drop(Price_Area_Outliers)

# 1. Convert boolean features to integers
boolean_features = ['Parking', 'Warehouse', 'Elevator']
dataset[boolean_features] = dataset[boolean_features].astype(int)

# 2. One-hot encode the Address feature
dataset["Address"]=dataset["Address"].apply(lambda x : re.sub('[^a-zA-Z]' ,'' ,x))
address_encoded = pd.get_dummies(dataset['Address'], prefix='Address')
dataset = pd.concat([dataset, address_encoded], axis=1)
dataset.drop('Address', axis=1, inplace=True)

prepared_dataset = dataset.drop(["Price(USD)"],axis=1)

# 3. Define features and target
X = prepared_dataset.drop(["Price"],axis=1)
y = prepared_dataset["Price"]

# 4. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the model
model = XGBRegressor() 
model.fit(X_train, y_train)

# 6. Save the model
joblib.dump(model, MODEL_PATH)

# 7. Evaluate the model
# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate R² scores
train_r2_score = r2_score(y_train, y_train_pred)
test_r2_score = r2_score(y_test, y_test_pred)

# Calculate Mean Squared Error and Mean Absolute Error
mse = mean_squared_error(y_test, y_test_pred)  # Use y_test_pred instead of y_pred
mae = mean_absolute_error(y_test, y_test_pred)

# Print the evaluation metrics
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Train R² Score: {train_r2_score:.2f}")
print(f"Test R² Score: {test_r2_score:.2f}")

print("Model training complete and saved to:", MODEL_PATH)