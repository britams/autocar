from pathlib import Path

# pip install seaborn
# pip install openpyxl
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    BASE = Path(__file__).parent
    df = pd.read_csv(BASE / "weather.csv", index_col=0, header=0)
    print(df)
    df = sns.load_dataset("titanic")
    df.to_csv(BASE / "titanic.csv")
    df_excell = pd.read_excel(BASE / "감귤평점테스트.xlsx")
    print(df_excell.info())


if __name__ == "__main__":
    main()