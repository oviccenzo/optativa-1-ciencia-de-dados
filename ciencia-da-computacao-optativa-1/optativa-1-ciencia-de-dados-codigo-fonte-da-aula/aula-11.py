
# Open In Colab
#
# %%capture
# !pip install keras_tuner
# !pip install tensorflow


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import json
import keras_tuner as kt
from tensorflow.keras.optimizers import SGD


# Carregar os dados do dataset
iris = load_iris()
X = iris.data
y = to_categorical(iris.target, num_classes=3)
target_names = iris.target_names


# Pré-processamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# Função para criar o modelo com hiperparâmetros variáveis
def construir_modelo(hp):
    model = Sequential()
    # Entrada
    model.add(Input(shape=(4,))) # 4 é o número de colunas do dataset

    units = hp.Int(f'units', min_value=4, max_value=64, step=4)
    model.add(Dense(units, activation='relu'))

    # Saída
    model.add(Dense(3, activation='softmax'))

    # Hiperparâmetros do otimizador
    lr = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4]) # 3 valores
    momentum = hp.Float('momentum', min_value=0.0, max_value=0.9, step=0.1) # 10 valores

    optimizer = SGD(learning_rate=lr, momentum=momentum)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Configurar a busca com o Keras Tuner
tuner = kt.RandomSearch(
    construir_modelo,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='kt_iris_tuning',
    project_name='iris_classification'
)

# Executar a busca
tuner.search(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)

# Resultados da melhor combinação
melhor_modelo = tuner.get_best_models(num_models=1)[0]
melhores_hp = tuner.get_best_hyperparameters(1)[0]
print(f'Melhores hiperparâmetros encontrados: ')
print(f' - Neurônios na camada escondida: {melhores_hp.get("units")}')
print(f' - Learning rate: {melhores_hp.get("learning_rate")}')
print(f' - Momentum: {melhores_hp.get("momentum")}')

# Avaliar no conjunto de teste
loss, acc = melhor_modelo.evaluate(X_test, y_test, verbose=0)
# print(f'\nAcurácia no conjunto de teste: {acc:.2f}')
# /usr/l o cal/l i b/p y thon3.1
# 1/d i st-p a ckages/k e ras/s r c/s a ving/s a ving_lib.py:75 7: UserWarning: Skipping v
# riable l
# ading f
# r optimizer 'SGD', because it has 2 variables whereas the saved optimizer has 6 variables.
# saveable.load_own_variables(weights_store.get(inner_path))
#
# Melhores h
# perparâmetros e
# contrados:
# - Neurônios na c
# mada e
# condida: 56
# - Learning r
# te: 0.01
# - Momentum: 0.8
#
# Acurácia no c
# njunto de t
# ste: 1.00

# Resultados da melhor combinação
melhor_modelo = tuner.get_best_models(num_models=1)[0]
melhores_hp = tuner.get_best_hyperparameters(1)[0]
print(f'Melhores hiperparâmetros encontrados: ')
print(f' - Neurônios na camada escondida: {melhores_hp.get("units")}')
print(f' - Learning rate: {melhores_hp.get("learning_rate")}')
print(f' - Momentum: {melhores_hp.get("momentum")}')

# Avaliar no conjunto de teste
loss, acc = melhor_modelo.evaluate(X_test, y_test, verbose=0)
print(f'\nAcurácia no conjunto de teste: {acc:.2f}')

# Melhores h
# perparâmetros e
# contrados:
# - Neurônios na c
# mada e
# condida: 56
# - Learning r
# te: 0.01
# - Momentum: 0.8
#
# Acurácia no c
# njunto de t
# ste: 1.00
#
# # Relatório
# tuner.results_summary()
#
# Results s
# mmary
# Results in kt_iris_tuning/i r is_classification
# Showing 10 b
# st t
# ials
# Objective(name="val_accuracy", direction="max")
#
# Trial 02 s
# mmary
# Hyperparameters:
# units: 56
# learning_rate: 0.01
# momentum: 0.8
# Score: 0.9583333134651184
#
# Trial 01 s
# mmary
# Hyperparameters:
# units: 20
# learning_rate: 0.01
# momentum: 0.6000000000000001
# Score: 0.9166666865348816
#
# Trial 08
# su
# mary
# Hyperparameters:
# units: 20
# learning_rate: 0.01
# momentum: 0.0
# Score: 0.875
#
# Trial 05
# su
# mary
# Hyperparameters:
# units: 44
# learning_rate: 0.0001
# momentum: 0.8
# Score: 0.8333333134651184
#
# Trial 03
# su
# mary
# Hyperparameters:
# units: 60
# learning_rate: 0.001
# momentum: 0.0
# Score: 0.7083333134651184
#
# Trial 06
# su
# mary
# Hyperparameters:
# units: 36
# learning_rate: 0.001
# momentum: 0.1
# Score: 0.7083333134651184
#
# Trial 09
# s
# u
# mary
# Hyperparameters:
# units: 40
# learning_rate: 0.001
# momentum: 0.4
# Score: 0.5
#
# Trial 07
# su
# mary
# Hyperparameters:
# units: 16
# learning_rate: 0.001
# momentum: 0.2
# Score: 0.4166666567325592
#
# Trial 04
# su
# mary
# Hyperparameters:
# units: 8
# learning_rate: 0.001
# momentum: 0.30000000000000004
# Score: 0.3333333432674408
#
# Trial 00
# su
# mary
# Hyperparameters:
# units: 20
# learning_rate: 0.0001
# momentum: 0.6000000000000001
# Score: 0.2916666567325592
#
