from pathlib import Path

import numpy as np


def main():
    BASE = Path(__file__).parent
    s1 = np.random.randint(0, 10, 100, dtype=np.int8).reshape(2, -1)
    s2 = s1[1, :]
    s1 = s1[0, :]
    # np.save(BASE / "randoms", s1)   # 208B -> 40B 138B 228B
    np.savez(BASE / "randomsz", arr1=s1, arr2=s2)
    np.savetxt(BASE / "randoms.txt", s1, fmt="%i")  # 사람이 볼 수 있게.


if __name__ == "__main__":
    main()