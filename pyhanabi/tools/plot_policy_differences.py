import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt


def plot_data():
    sad = 13
    seq_len = 70

    br = genfromtxt(f'similarity_data/br/br_sad_1_3_6_7_8_12_vs_sad_{sad}.csv', delimiter=',')
    sba = genfromtxt(f'similarity_data/sba/sba_sad_1_3_6_7_8_12_vs_sad_{sad}.csv', delimiter=',')

    time_steps = list(range(seq_len))

    plt.plot(time_steps, br[:seq_len], label="br", color="green")
    plt.plot(time_steps, sba[:seq_len], label="sba", color="blue")
    plt.legend(loc="best")
    # plt.title(f"SAD_{sad} Policy Similarities")
    plt.title(f"OBL1 Policy Similarities")
    plt.xlabel("Time Step")
    plt.ylabel(f"Similarity vs OBL1")
    plt.show()

if __name__ == "__main__":
    plot_data()
