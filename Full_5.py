import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, RNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from LTC5 import LTCCell  # Assuming LTCCell is correctly imported

# Load and prepare the data
df = pd.read_excel('capacityFade.xlsx')
df = df[['Cap_2000', 'Cap_3500']].dropna()


scaler = MinMaxScaler(feature_range=(0, 1))
data_2000_scaled = scaler.fit_transform(df['Cap_2000'].values.reshape(-1, 1))

def create_dataset(dataset, look_back=5):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X).reshape(-1, look_back, 1), np.array(Y)

look_back = 5
train_size = 1950  # Adjusted train size
X, Y = create_dataset(data_2000_scaled, look_back)

X_train, Y_train = X[:train_size], Y[:train_size]
X_test, Y_test = X[train_size:], Y[train_size:]


print(X_train.shape)
'''
# Define and train the model
model = Sequential([
    RNN(LTCCell(10), input_shape=(look_back, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, Y_train, epochs=50, batch_size=1, verbose=1)

# Save and load model
model.save('LTC_model.keras')
model = load_model('LTC_model.keras', custom_objects={'LTCCell': LTCCell})

# Predictions and MSE calculations
predictions = model.predict(X_test)
mse_train = mean_squared_error(Y_train, model.predict(X_train))
mse_test = mean_squared_error(Y_test, predictions)

# Visualization and other results processing as required

print(f"Training MSE: {mse_train}")
print(f"Test MSE: {mse_test}")

'''