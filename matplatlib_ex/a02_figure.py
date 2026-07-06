# pip install matplotlib

import matplotlib.pyplot as plt
import numpy as np


def main():
    data1 = np.random.random(10)
    data2 = np.random.random(30).reshape(10, 3)
    # fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    # axes[0].plot(data1)
    # axes[1].plot(data2)
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(2, 1, 1)      #(가로 세로 격자), 인덱스(순서)
    ax1.plot(data1)
    ax1.set_title("sensor 1")
    ax1.set_xlabel("Index(cm)")
    ax1.set_ylabel("Count(EA)")
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(data2, label=["sensor 1", "sensor 2", "sensor 2"])
    ax2.legend(loc="upper right")
    fig.show()
    input("Enter")
    # plt.show()


if __name__ == "__main__":
    main()