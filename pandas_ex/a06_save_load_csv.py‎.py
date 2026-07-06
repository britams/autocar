from pathlib import Path

import numpy as np
import pandas as pd


def main():
    BASE = Path(__file__).parent
    value = [[32, 68, 220, 72],
             [28, 30, 0, 12],
             [38, 81, 0, 91]]
    columns = ["온도", "습도", "강수량", "불쾌지수"]
    index = ["초여름", "늦봄", "한여름"]
    df = pd.DataFrame(value, index=index, columns=columns, dtype=np.uint8)

    df.to_csv(BASE / "weather.csv")


if __name__ == "__main__":
    main()