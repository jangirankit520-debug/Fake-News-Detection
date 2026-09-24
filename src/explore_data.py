import pandas as pd
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "Data"

fake = pd.read_csv(data_dir / "Fake.csv")
true = pd.read_csv(data_dir / "True.csv")

print("Fake shape:", fake.shape)
print("True shape:", true.shape)
print("Columns:", fake.columns.tolist())

fake["label"]=0
true['label']=1
data=pd.concat([fake,true],ignore_index=True)

print("\nClass balance:\n",data["label"].value_counts())

print("\nMissing values:\n",data.isnull().sum())
print("Duplicate texts:",data.duplicated(subset="text").sum())
print("Empty texts:",(data["text"].str.strip()=="").sum())

print("\nFake subjects:\n",fake["subject"].value_counts())
print("\nTrue subjects:\n",true["subject"].value_counts())

data["word_count"]=data["text"].str.split().str.len()
print("\nWord count by label:\n",data.groupby("label")["word_count"].describe())

print("\nTrue starts:\n",true["text"].head(5).str[:120].to_string())
print("\nFake starts:\n", fake["text"].head(5).str[:120].to_string())















