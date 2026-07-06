import numpy as np
import pandas as pd


def main():
    value = [[32, 68, 220, 72],
             [28, 30, 0, 12],
             [38, 81, 0, 91]]
    columns = ["온도", "습도", "강수량", "불쾌지수"]
    index = ["초여름", "늦봄", "한여름"]
    df = pd.DataFrame(value, index=index, columns=columns, dtype=np.uint8)

    print(df.drop("초여름", inplace=True))        # return 객체에만 반영
    print(df.drop("온도", axis=1))
    print(df.drop(columns="강수량"))

    df["불쾌지수"]["한여름"] = np.nan
    print(df.replace(0, np.nan))
    print(df)


if __name__ == "__main__":
    main()