from pathlib import Path

import numpy as np


def main():
    BASE = Path(__file__).parent
    s1 = np.load(BASE / "randoms.npy")
    print(s1)
    file = np.load(BASE / "randomsz.npz")
    print(file["arr1"])
    print(file["arr2"])
    file.close()
    y = np.loadtxt(BASE / "randoms.txt", dtype=np.int8)
    print(y, type(y))


if __name__ == "__main__":
    main()