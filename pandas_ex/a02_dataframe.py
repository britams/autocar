# iloc[0,1] 이것은 넘파이에서 판다스 접근하는 방식.
# 넘파이는 수학과 가깝다면 판다스는 데이터 위주. 
# 판다스는 파이선 버전 엑셀. 데이타 프레임은 엑셀 시트 
# pip install pandas

import numpy as np
import pandas as pd


def main():
    value = [[32, 68, 220, 72],
             [28, 30, 0, 12],
             [38, 81, 0, 91]]
    columns = ["온도", "습도", "강수량", "불쾌지수"]
    index = ["초여름", "늦봄", "한여름"]
    df = pd.DataFrame(value, index=index, columns=columns, dtype=np.uint8)
    print(df)
    print(df["온도"]["늦봄"], df.iloc[1, 0], df.iloc[1][0])
    print(df.iloc[1:3, 1:3])
    print(df.index, df.columns, df.values)
    print(type(df.index), type(df.columns), type(df.values))

    print(df.info())
    print(df.head(2))
    # 데이터를 불러올때 (빅데이터) 불러오는 방식! excel, csv(comma seperate value)- tsv..


if __name__ == "__main__":
    main()