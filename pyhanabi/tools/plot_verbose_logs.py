import argparse
from parse_verbose_logs import parse_logs
import matplotlib
import matplotlib.pyplot as plt


def plot_data(filename):
    data = parse_logs(filename, float("inf"))
    for key, value in data.items():
        create_plot(data, key)


def create_plot(data, y_data):
    plt.plot(data["epoch"], data[y_data])
    plt.xlabel("epoch")
    plt.ylabel(y_data)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True)
    args = parser.parse_args()

    plot_data(args.filename)
