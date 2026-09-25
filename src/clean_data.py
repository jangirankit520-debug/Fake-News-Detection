import pandas as pd
import re
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "Data"

def load_raw_data():
    fake=pd.read_csv(data_dir/"Fake.csv")
    true=pd.read_csv(data_dir/"True.csv")
    fake["label"]=0
    true["label"]=1
    data=pd.concat([fake,true],ignore_index=True)
    return data

def basic_clean(data):
    data=data.drop_duplicates(subset="text")
    data=data[data["text"].str.strip() !=""]
    data=data.dropna(subset=["text"])
    return data

def remove_leakage_clues(text):
    text = re.sub(r'^[A-Z\s,/]+\((?:Reuters|AP)\)\s*-\s*', '', text)
    text = re.sub(r'\((?:Reuters|AP)\)', '', text)
    return text

if __name__=="__main__":
    data=load_raw_data()
    print("Before cleaning: ",data.shape)

    data=basic_clean(data)
    print("After removing duplicates/empty rows: ",data.shape)

    data["text_no_leak"]=data["text"].apply(remove_leakage_clues)
    print("\nExample before: \n",data["text"].iloc[0][:150])
    print("\nExample after:\n", data["text_no_leak"].iloc[0][:150])

    real_example = data[data["label"]==1].iloc[0]
    print("\nReal article before:\n", real_example["text"][:150])
    print("\nReal article after:\n", real_example["text_no_leak"][:150])







