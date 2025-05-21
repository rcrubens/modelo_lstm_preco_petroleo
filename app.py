import warnings
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; margin-top: -20px; '>Modelo LSTM - Previsão de Preços de Petróleo</h1>", unsafe_allow_html=True)

st.sidebar.header("Parâmetros do Modelo")
prediction_ahead = st.sidebar.number_input("Quantidade de dias:", min_value=1, max_value=30, value=15, step=1)

if "model" not in st.session_state:
    st.session_state.model = None

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
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, batch_size=1, epochs=3, verbose=1, callbacks=[early_stopping])

    st.session_state.model = model

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_predictions = scaler.inverse_transform(train_predictions)
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))

    test_predictions = scaler.inverse_transform(test_predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    last_60_days = data_train_scaled[-seq_length:]
    future_input = last_60_days.reshape(1, seq_length, 1)
    future_forecast = []

    for _ in range(prediction_ahead):
        next_pred = model.predict(future_input)[0, 0]
        future_forecast.append(next_pred)
        next_input = np.append(future_input[0, 1:], [[next_pred]], axis=0)
        future_input = next_input.reshape(1, seq_length, 1)

    future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1))

    train_range = df_preco.index[seq_length:train_size]
    test_range = df_preco.index[train_size:train_size + len(test_predictions)]
    future_index = pd.date_range(start=df_preco.index[-1], periods=prediction_ahead + 1, freq='D')[1:]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_preco.index, y=df_preco['preco'], 
                             mode='lines', name='Preços reais', 
                             line=dict(color='blue')))

    fig.add_trace(go.Scatter(x=[df_preco.index[train_size]] * 2, y=[min(df_preco['preco']), max(df_preco['preco'])], 
                             mode='lines', name='Train/Test Split', 
                             line=dict(color='gray', dash='dash')))

    fig.add_trace(go.Scatter(x=train_range, y=train_predictions[:len(train_range)], 
                             mode='lines', name='Train Predictions', 
                             line=dict(color='green')))

    fig.add_trace(go.Scatter(x=test_range, y=test_predictions[:len(test_range)], 
                             mode='lines', name='Test Predictions', 
                             line=dict(color='orange')))

    fig.add_trace(go.Scatter(x=future_index, y=future_forecast.flatten(), 
                             mode='lines', name=f'{prediction_ahead}-Day Forecast', 
                             line=dict(color='red')))

    fig.update_layout(title='Modelo LSTM - Previsão de Preços de Petróleo',
                      xaxis_title='Data',
                      yaxis_title='Preço (USD)',
                      template='plotly_white',
                      legend_title="Legenda",
                      width=1000, height=500)

    st.plotly_chart(fig)
