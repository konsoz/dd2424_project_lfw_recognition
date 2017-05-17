import matplotlib.pyplot as plt
import numpy as np



def read_file(file_name, delimiter_param=','):
    data_array = np.genfromtxt(file_name, delimiter=delimiter_param, skip_header=True)
    return data_array


def plot_cost(costs_train,costs_validation, n_epochs):
    epochs_arr = np.arange(0, n_epochs).tolist()

    plt.plot(epochs_arr, costs_train, 'r-',label='training loss')
    plt.plot(epochs_arr, costs_validation, 'b-',label='validation loss')
    plt.legend(loc='upper left', shadow=True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()



file_name1 = "data/run_train,tag_accuracy.csv"
file_name2 = "data/run_valid,tag_accuracy.csv"

file1 = read_file(file_name1)
file2 = read_file(file_name2)

plot_cost(file1[:,2], file2[:,2], 30)