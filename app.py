import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.markdown("<style>.main {padding-top: @px; }</style>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; margin-top: -20px; '>Modelo LSTM - Previsão de Preços de Petróleo</h1>", unsafe_allow_html=True)

st.sidebar.header("Parâmetros do Modelo")

prediction_ahead = st.sidebar.number_input("Quantidade de dias:", min_value=1, max_value=30, value=15, step=1)

if st.sidebar.button("Calcule"):

    df_preco = pd.read_csv('ipeadata_petroleo.csv')
    df_preco['data'] = pd.to_datetime(df_preco['data'])
    df_preco['preco'] = df_preco['preco'].ffill()
    df_preco.set_index('data', inplace=True)

    split = int(len(df_preco['preco']) * 0.7)
    treino, teste = df_preco['preco'][:split], df_preco['preco'][split:]

    train_size = len(treino)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_train_scaled = scaler.fit_transform(treino.values.reshape(-1, 1))
    data_test_scaled = scaler.transform(teste.values.reshape(-1, 1))

    def create_sequences(data, seq_length=1):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i+seq_length), 0])
            y.append(data[i+seq_length, 0])
        return np.array(X), np.array(y)
    
    seq_length = 15
    X_train, y_train = create_sequences(data_train_scaled, seq_length)
    X_test, y_test = create_sequences(data_test_scaled, seq_length)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(50, activation='relu', return_sequences=False),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.add(Dropout(0.2))  # 20% dos neurônios desativados aleatoriamente para prevenir o overfitting

    # early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, batch_size=1, epochs=3, verbose=1) # callbacks=[early_stopping]

    train_predictions = model.predict(X_train)
    test_predictions = model. predict(X_test)

    train_predictions = scaler.inverse_transform(train_predictions)
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))

    test_predictions = scaler.inverse_transform(test_predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    last_60_days = data_train_scaled[-seq_length:]
    future_input = last_60_days.reshape(1, -seq_length, 1)
    future_forecast = []

    for _ in range(prediction_ahead):
        next_pred = model.predict(future_input)[0, 0]
        future_forecast.append(next_pred)
        next_input = np.append(future_input[0, 1:], [[next_pred]], axis=0)
        future_input = next_input.reshape(1, seq_length, 1)

    future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1))

    plt.figure(figsize=(14, 5))
    plt.plot(df_preco.index, df_preco['preco'], label='Preços reais', color='blue')
    plt.axvline(x=df_preco.index[train_size], color='gray', linestyle='--', label='Train/Test Split' )

    train_range = df_preco.index[seq_length:train_size]
    test_range = df_preco.index[train_size:train_size + len(test_predictions)]
    plt.plot(train_range, train_predictions[:len(train_range)], label='Train Predictions', color='green' )
    plt.plot(test_range, test_predictions[:len(test_range)], label='Test Predictions', color='orange')

    future_index = pd.date_range(start=df_preco.index[-1], periods=prediction_ahead + 1, freq='D') [1:]
    plt.plot(future_index, future_forecast, label=f'{prediction_ahead}-Day Forecast', color='red')

    plt.title('Modelo LSTM')
    plt.xlabel('Data')
    plt.ylabel('Preço (USD)')
    plt.legend()
    plt.show()
