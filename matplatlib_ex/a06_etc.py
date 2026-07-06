import matplotlib.pyplot as plt
import numpy as np


def main():
    data = np.random.random(10)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(2, 3, 1)
    ax.hist(data)
    ax = fig.add_subplot(2, 3, 2)
    ax.pie(data)
    ax = fig.add_subplot(2, 3, 3)
    ax.step(data, data)
    ax = fig.add_subplot(2, 3, 4)
    ax.boxplot(data)    # 중요 이상치(outlier) 결측치.
    ax = fig.add_subplot(2, 3, 5)
    ax.violinplot(data)
    ax = fig.add_subplot(2, 3, 6)
    X, Y = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
    Z = np.sin(X) + np.cos(Y)
    ax.contour(X, Y, Z)
    plt.show()


if __name__ == "__main__":
    main()