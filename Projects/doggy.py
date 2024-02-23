import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Loading the data
df = pd.read_csv('C:/Users/achit/OneDrive/Desktop/data/Dog Intelligence.csv', encoding='ISO-8859-1')

# ETA

# Background Checks

initial_checks = {
    "head": df.head(),
    "shape": df.shape,
    "info": df.info(),  # This will print out info, not return
    "null_values": df.isnull().sum(),
    "dtypes": df.dtypes,
    "duplicates": df.duplicated().sum(),
    "describe": df.describe()
}

initial_checks

# Checking for null values
df.isnull().sum()

# Checking the missing values for what data type it is

df.dtypes

# Checking for Duplicates

df.duplicated().sum()

# Checking for Outliers

df.describe()

# Imputing missing values for 'reps_lower' and 'reps_upper' with the mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df[['reps_lower', 'reps_upper']] = imputer.fit_transform(df[['reps_lower', 'reps_upper']])

# Combining 'reps_lower' and 'reps_upper' into a new 'reps' column as their average
df['reps'] = (df['reps_lower'] + df['reps_upper']) / 2

# Separating features and target variable before encoding and train-test split
X = df.drop(columns=['reps', 'reps_lower', 'reps_upper'])  # Dropping original 'reps' columns to avoid leakage
y = df['reps']

# Encoding the 'Breed' column
le = LabelEncoder()
X['Breed'] = le.fit_transform(X['Breed'])

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Calculating the mean absolute error (in thousands) as an evaluation metric
mae = (y_pred - y_test).abs().mean() * 1000

mae



coefficients = pd.DataFrame(model.coef_, X_train.columns, columns=['Coefficient'])

coefficients.loc[['height_low_inches', 'height_high_inches', 'weight_low_lbs', 'weight_high_lbs']]


print( "These coefficients indicate how a unit increase in each feature affects the learning ability 'reps', "
       "with all other variables held constant. The negative coefficients suggest that, according to the model, "
       "an increase in both height and weight is associated with a slight decrease in learning ability "
       "(increase in 'reps', meaning more repetitions needed to learn). This could imply that, to some extent, "
       "larger dogs (in terms of both height and weight) may require more repetitions to learn compared to smaller dogs.")


r_squared = model.score(X_test, y_test)

r_squared

print("The R-squared value for the linear regression model, when applied to the test set, "
      "is approximately 0.486. This value indicates that around 48.6% of the variance in the"
      " learning ability ('reps') can be explained by the model's predictors, including the "
      "size-related features (height and weight), among others.")




# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Decision Tree model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Predicting on the test set
y_pred_dt = dt_model.predict(X_test)

# Evaluating the model
r2_score_dt = r2_score(y_test, y_pred_dt)

r2_score_dt

print("The Decision Tree model has been successfully trained and evaluated on the dataset."
      "The R-squared value for the Decision Tree model on the test set is approximately 0.576. "
      "This indicates that around 57.6% of the variance in the learning ability ('reps') "
      "is explained by the model's predictors.)

# Using Keras Model to make predictions

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Building the model

model = Sequential()
model.add(Dense(64, input_dim = X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='softmax'))
model.add(Dense(1, activation='linear'))

# Combile the model

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Train the model

model.fit(X_train, y_train, epochs = 50, validation_data = (X_test, y_test), verbose = 2, batch_size = 10)

# Evaluating the model

scores = model.evaluate(X_test, y_test, verbose=0)

print(scores)


