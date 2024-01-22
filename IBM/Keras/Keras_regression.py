import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()

concrete_data.shape

concrete_data.describe()

concrete_data.isnull().sum()

concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

predictors.head()

target.head()
# We Could either use this or Standard Scaler
predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()
 # OR
scaler = StandardScaler()
predictors_norm = scaler.fit_transform(predictors)

n_cols = predictors_norm.shape[1] # number of predictors


# defining the model

def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = regression_model()

model.fit(predictors_norm, target, epochs=1000, verbose = 2)

