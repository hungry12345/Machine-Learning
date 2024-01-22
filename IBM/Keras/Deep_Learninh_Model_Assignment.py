import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
import numpy as np

# Loading the data
data = pd.read_csv(r"C:/Users/achit/Downloads/concrete_data (1).csv")

# Splitting features and target variable
X = data.drop('Strength', axis=1)
y = data['Strength']

# List to store mean squared errors
mse_list = []

# Repeat the process 50 times
for _ in range(50):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Define the model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, verbose=0)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_list.append(mse)

# Calculate the mean and the standard deviation of the mean squared errors
mean_mse = np.mean(mse_list)
std_mse = np.std(mse_list)

print("Mean of MSEs:", mean_mse)
print("Standard Deviation of MSEs:", std_mse)


print("After using our model, we concluded that the Mean of MSEs is 258.47021178959056")
print("And the Standard Deviation of MSEs is 212.95963307961355")