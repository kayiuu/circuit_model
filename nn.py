import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from regression import read_data
from regression import plot_predict_vs_actual


def rnn(batch_size=100, time_steps=10, features=4):
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Masking(mask_value=0, batch_input_shape=(batch_size, time_steps, features)),
    #     tf.keras.layers.LSTM(time_steps, return_sequences=True, stateful=True),
    # ])
    model = tf.keras.models.Sequential([
        tf.keras.layers.Masking(mask_value=0, batch_input_shape=(batch_size, time_steps, features)),
        tf.keras.layers.LSTM(time_steps, return_sequences=True, stateful=True),
        tf.keras.layers.Dense(1)
    ])
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Masking(mask_value=0, batch_input_shape=(1000, 1, features)),
    #     tf.keras.layers.LSTM(8, stateful=True),
    #     #tf.keras.layers.Dense(16, activation='relu'),
    #     #tf.keras.layers.Dense(10, activation='relu'),
    #     tf.keras.layers.Dense(1, activation='relu'),
    # ])
    return model


def simple_nn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model


def cnn(timesteps=10):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=16, kernel_size=timesteps, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model


def zero_pad(x, y, batch_size=100, time_steps=10):
    input_len = batch_size * time_steps
    features = x.shape[1]
    x_rnn = np.append(x, np.zeros((input_len - x.shape[0], features)), axis=0)
    x_rnn = np.reshape(x_rnn, (batch_size, time_steps, features))
    y_rnn = np.append(y, np.zeros(input_len - x.shape[0]))
    y_rnn = np.reshape(y_rnn, (batch_size, time_steps, 1))
    return x_rnn, y_rnn


def main():
    _, x_train, y_train = read_data("sim_data_train.csv")
    t, x_test, y_test = read_data("sim_data_test.csv")

    nn_type = rnn

    if nn_type == rnn or nn_type == cnn:
        batch_size = 100
        time_steps = 10
        x_train_processed, y_train_processed = zero_pad(x_train, y_train, batch_size=batch_size, time_steps=time_steps)
        x_test_processed, y_test_processed = zero_pad(x_test, y_test, batch_size=batch_size, time_steps=time_steps)
    else:
        x_train_processed, y_train_processed = x_train, y_train
        x_test_processed, y_test_processed = x_test, y_test

    model = nn_type()
    model.summary()
    loss_fn = tf.keras.losses.MeanSquaredError(name='mean_squared_error')
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.fit(x_train_processed, y_train_processed, epochs=20)
    predictions = model(x_test_processed).numpy()
    if nn_type == rnn or nn_type == cnn:
        predictions = predictions.reshape(-1)[:y_test.shape[0]]
    #TODO: debug CNN and RNN
    plot = True
    if plot:
        plot_predict_vs_actual(time=t, vdd=x_test[:, 1], inp=x_test[:, 2], inn=x_test[:, 3], y_predict=predictions,
                               y=y_test)
    print("loss = ", loss_fn(y_test.reshape(-1, 1), predictions).numpy())


if __name__ == "__main__":
    main()