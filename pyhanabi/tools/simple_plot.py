import matplotlib.pyplot as plt
import numpy as np

def plot_data():
    time_step = list(range(len(br)))
    plt.plot(time_step, br, label="br", color="green")
    plt.plot(time_step, sba, label="sba", color="blue")
    plt.legend(loc="best")
    plt.title("SAD_5 Policy Similarities")
    plt.xlabel("Similarity vs SAD_5")
    plt.ylabel("Time Step")
    plt.show()

if __name__ == "__main__":
    plot_data()
