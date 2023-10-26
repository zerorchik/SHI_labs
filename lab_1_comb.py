import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from seaborn import pairplot
# Ігнорування warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --------------------------------------Допоміжний блок--------------------------------------

# Функція побудови моделі linear
def build_and_compile_model_linear(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(units=1)
    ])
    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.1))
    return model

# Функція побудови моделі DNN
def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model

# Функція історії абсолютних помилок
def learning_hist(model, epoch):
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        verbose=0,
        epochs=epoch
    )
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail(), '\n')
    return history

# Функція оцінки моделі на валідаційних даних
def val_loss(model):
    val_loss = model.evaluate(X_val, y_val)
    print('Валідаційна абсолютна помилка:', val_loss, '\n')
    return val_loss

# Графік функції втрат
def plot_loss(history, title):
    plt.plot(history.history['loss'], label='Тренувальна')
    plt.plot(history.history['val_loss'], label='Тестова')
    plt.ylim([0, 5])
    plt.xlabel('Епохи')
    plt.ylabel('Помилка оцінки якості вина')
    plt.legend()
    plt.grid(True)
    plt.title(title)

# ---------------------------------------Основний блок---------------------------------------

files = ['winequality-white.csv', 'winequality-red.csv']

# Вибір режиму зчитування даних
print('Оберіть джерело вхідних даних та подальші дії:')
for i in range(len(files)):
    print(i + 1, '-', files[i])
data_mode = int(input('mode:'))

# Якщо джерело даних існує
if data_mode in range(1, len(files) + 1):
    # Завантаження датасету
    file = files[data_mode - 1]
    data = pd.read_csv(file, sep=';')
    # Вигляд датасету
    print('-----------------------------')
    print(file, ':')
    print('-----------------------------')
    print(data.head(5), '\n')

    # Розділ датасету на ознаки - результати тестів вина (X) та цільову змінну - якість вина (y)
    X = data.drop('quality', axis=1)
    y = data['quality']
    # Розділ даних на тренувальний (75%), тестовий (15%) і валідаційний (10%) набори
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)
    # Перевірка співвідношення 75% : 15% : 10%
    print('Ознак тренувального датасету:', X_train.shape)
    print('Ознак тестового датасету:', X_test.shape)
    print('Ознак валідаційного датасету:', X_val.shape, '\n')

    # Опис датасету
    print('-----------------------------')
    print('Опис:')
    print('-----------------------------')
    print(data.describe().transpose(), '\n')
    # # Побудова графіків pairplot
    # pairplot(data[data.columns], diag_kind='kde')
    # plt.show()
    # Нормалізація ознак
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(X_train))

# ----------------------Лінійна регресійна модель з декількома ознаками----------------------

    # Лінійна модель
    linear_model = build_and_compile_model_linear(normalizer)
    # Опис моделі
    linear_model.summary()
    print('')
    # Навчання моделі
    history = learning_hist(linear_model, 50)
    # Функція втрат
    linear_loss = val_loss(linear_model)
    plot_loss(history, 'Лінійна модель')
    plt.show()
    # Збереження результатів
    test_results = {}
    test_results['linear_model'] = linear_loss

# ----------------------Глибока нейромережа (DNN) з декількома ознаками----------------------

    # Глибока нейромережа (DNN)
    dnn_model = build_and_compile_model(normalizer)
    # Опис моделі
    dnn_model.summary()
    print('')
    # Навчання моделі
    history = learning_hist(dnn_model, 100)
    # Функція втрат
    dnn_loss = val_loss(dnn_model)
    plot_loss(history, 'DNN модель')
    plt.show()
    # Збереження результатів
    test_results['dnn_model'] = dnn_loss
    # Порівняння моделей
    print(pd.DataFrame(test_results, index=['Середня абсолютна помилка оцінки якості вина']).T)