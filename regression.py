import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def read_data(file_name):
    data = np.genfromtxt(file_name, delimiter=',')
    t = data[1:, 0]
    x = data[1:, 1:]
    y = data[1:, -1]
    return t, x, y


def train_regression(state, kernel, learning_rate, x, y):
    n = x.shape[0]
    for i in range(n):
        update_state(state, kernel, learning_rate, x[i], y[i])


def update_state(state, kernel, learning_rate, x_i, y_i):
    prediction = predict(state, kernel, x_i)
    beta_i = learning_rate * (y_i - prediction)
    state.append((beta_i, x_i))


def dot_kernel(a, b):
    return np.dot(a, b)


def poly_kernel(a, b, degree=3):
    sum = 0
    for i in range(degree+1):
        sum += np.dot(a, b) ** i
    return sum


def poly_tanh_kernel(a, b, degree=3):
    sum = 0
    a_extended = np.append(a, np.tanh(a))
    b_extended = np.append(b, np.tanh(b))
    for i in range(degree+1):
        sum += np.dot(a_extended, b_extended) ** i
    return sum


def predict(state, kernel, x):
    sum = 0
    for weight, x_i in state:
        sum += weight * kernel(x, x_i)
    return sum


def plot_predict_vs_actual(time, vdd, inp, inn, y_predict, y):
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(time, vdd, label='vdd')
    ax1.legend(loc="upper right")
    ax2.plot(time, inp - inn, label='inp-inn')
    ax2.legend(loc="upper right")
    ax3.plot(time, y_predict, 'r', label='predicted output')
    ax3.plot(time, y, 'b', label='actual output')
    ax3.legend(loc="upper right")
    plt.ylabel("voltage (V)")
    plt.xlabel("time (s)")
    plt.show()


def calc_error(y_predict, y):
    return np.sum((y_predict-y)**2)


def main():
    state = []
    kernel = poly_tanh_kernel
    _, x_train, y_train = read_data("sim_data_train.csv")
    train_regression(state, kernel, 1e-3, x_train, y_train)

    t, x_test, y_test = read_data("sim_data_test.csv")
    y_predict = [predict(state, kernel, x_test[i, :]) for i in range(y_test.shape[0])]

    plot = True
    if plot:
        plot_predict_vs_actual(time=t, vdd=x_test[:, 1], inp=x_test[:, 2], inn=x_test[:, 3], y_predict=y_predict, y=y_test)

    loss_fn = tf.keras.losses.MeanSquaredError(name='mean_squared_error')
    print("loss = ", loss_fn(y_test.reshape(-1, 1), y_predict).numpy())


if __name__ == "__main__":
    main()