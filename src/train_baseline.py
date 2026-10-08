from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression,PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

from clean_data import load_raw_data,basic_clean

data=load_raw_data()
data=basic_clean(data)

x=data["text"]
y=data["label"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42,stratify=y)

vectorizer=TfidfVectorizer(stop_words="english",max_df=0.7)
xv_train=vectorizer.fit_transform(x_train)
xv_test=vectorizer.transform(x_test)

models={"LogisticRegression":LogisticRegression(max_iter=1000),"Naive Bayes":MultinomialNB(),"Passive Aggressive":PassiveAggressiveClassifier(max_iter=50,random_state=42)}

result=[]
for name ,model in models.items():
    model.fit(xv_train,y_train)
    y_pred=model.predict(xv_test)

    acc=accuracy_score(y_test,y_pred)
    prec=precision_score(y_test,y_pred)
    rec=recall_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)

    result.append((name,acc,prec,rec,f1))

    print(f"\n---{name}---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 score:  {f1:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))















