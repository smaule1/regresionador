import pandas as pd

def clean_data():
    df = pd.read_csv("covid_data.csv")
    df.pop("iso_code")
    df.pop("continent")    
    df = df[df["location"] == "Japan"]
    df["date"] = pd.to_datetime(df["date"]).astype("int64") // (10**9 * 60 *60 * 24) - (50*365)
    df = df.fillna(0)
    print(df)
    df.to_csv("japan_data.csv", index=False)
    
