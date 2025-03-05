import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("../data/raw/all_cities.csv")
    print(df.head())
