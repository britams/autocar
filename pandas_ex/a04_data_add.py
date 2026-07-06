import numpy as np
import pandas as pd


def main():
    value = [[32, 68, 220, 72],
             [28, 30, 0, 12],
             [38, 81, 0, 91]]
    columns = ["온도", "습도", "강수량", "불쾌지수"]
    index = ["초여름", "늦봄", "한여름"]
    df = pd.DataFrame(value, index=index, columns=columns, dtype=np.uint8)

    value2 = [[37, 90, 120, 94]]
    index2 = ["한여름"]
    df2 = pd.DataFrame(value2, index=index2, columns=columns)

    # print(df._append(df2))
    print(pd.concat([df, df2], axis=0))     # return 객체를 변화시킴
    df.insert(0, "자외선", [6, 3, 7])       # inplace=True 기본
    print(df)
    print(df.sort_index(inplace=True))      # inplace 를 옵션으로 쓸 수 있다.
    # sort_df = df.sort_index()
    print(df)
    # dropna, fillna, replace, sort, ....


if __name__ == "__main__":
    main()