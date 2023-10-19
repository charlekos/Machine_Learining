import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import pickle

# Inicializa o MetaTrader 5
mt5.initialize()

# Especifique o símbolo e o período de tempo dos dados que você deseja baixar
symbol = "USDJPY"  # Substitua pelo símbolo desejado
timeframe = mt5.TIMEFRAME_M1  # Período de 1 minuto

# Baixe os dados históricos do MetaTrader 5
data = mt5.copy_rates_from_pos(symbol, timeframe, 0, 99000)  # Os últimos 5000 períodos de dados

# Encerre a conexão com o MetaTrader 5
mt5.shutdown()

# Salve os dados em um arquivo CSV
data_df = pd.DataFrame(data)
data_df['time'] = pd.to_datetime(data_df['time'], unit='s')
data_df.to_csv(f'{symbol}_dados_historicos2.csv', index=False)

# Defina os hiperparâmetros para a rede neural
input_features = ['open', 'close', 'tick_volume']
output_feature = 'close'

look_back = 1
split_ratio = 0.8

# Preparação dos dados
data_df['target'] = data_df[output_feature].shift(-1)
data_df = data_df[input_features + ['target']].dropna()

X = data_df[input_features].values
y = data_df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - split_ratio, shuffle=False)

# Crie e treine a rede neural com duas camadas ocultas adicionais
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(look_back, len(input_features))))
model.add(Dense(50, activation='relu'))  # Primeira camada oculta adicional
model.add(Dense(50, activation='relu'))  # Segunda camada oculta adicional
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Redimensione os dados de entrada
X_train = X_train.reshape(X_train.shape[0], look_back, len(input_features))
X_test = X_test.reshape(X_test.shape[0], look_back, len(input_features))

# Realimentação e Re-Treinamento
for _ in range(5):  # Repita o processo de realimentação e re-treinamento cinco vezes
    model.fit(X_train, y_train, epochs=10, batch_size=1)
    
    # Avaliação do Modelo
    y_pred = model.predict(X_test)
    
    # Coleta de Dados para Realimentação
    incorrect_predictions = (y_pred != y_test)

    # Use the boolean array to filter X_test
    realimentation_X = X_test[incorrect_predictions[:, 0]]
    realimentation_y = y_test[incorrect_predictions[:, 0]]
    
    # Re-Treinamento com Realimentação
    if len(realimentation_X) > 0:
        model.fit(realimentation_X, realimentation_y, epochs=10, batch_size=1)

# Salve o modelo treinado em um arquivo usando o módulo pickle
with open(f'{symbol}_modelo_rede_neural2.pkl', 'wb') as file:
    pickle.dump(model, file)
