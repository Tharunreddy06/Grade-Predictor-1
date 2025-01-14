import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset with correct delimiter
data = pd.read_csv('student-mat.csv', delimiter=';')

# Print column names to verify
print(data.columns)

# Select features and target variable
X = data[['studytime', 'failures', 'absences', 'G1', 'G2']]  # Update these if necessary
y = data['G3']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully!")
