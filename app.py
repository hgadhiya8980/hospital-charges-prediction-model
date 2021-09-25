from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("insurance.csv")

num_var = df.select_dtypes(["int64","float64"]).keys()
cat_cols = df.select_dtypes(["O"]).keys()
df_cols = df.columns

df2 = df["sex"]
df2 = pd.DataFrame({"sex":df2})
df3 = df["smoker"]
df3 = pd.DataFrame({"smoker":df3})
df4 = df["region"]
df4 = pd.DataFrame({"region":df4})

value = df["sex"].unique()
for num,var in enumerate(value):
    num+=1
    df["sex"].replace(var, num, inplace=True)

value1 = df["smoker"].unique()
for num,var in enumerate(value1):
    num+=1
    df["smoker"].replace(var, num, inplace=True)

value2 = df["region"].unique()
for num,var in enumerate(value2):
    num+=1
    df["region"].replace(var, num, inplace=True)

X = df.drop("charges", axis=1)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

sc=StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


model = joblib.load("hospital_charges_prediction_model.pkl")

def hospital_charges_prediction_model(model, age, sex, bmi, children, smoker, region):
    
    for num,var in enumerate(value):
        if var==sex:
            sex=num 
    for num,var in enumerate(value1):
        if var==smoker:
            smoker=num
    for num,var in enumerate(value2):
        if var==region:
            region=num
            
    x = np.zeros(len(X.columns))
    x[0] = age
    x[1] = sex
    x[2] = bmi
    x[3] = children
    x[4] = smoker
    x[5] = region
    
    x = sc.transform([x])[0]
    return model.predict([x])[0]

app=Flask(__name__)

@app.route("/")
def home():
    value7 = list(df2["sex"].value_counts().keys())
    value7.sort()
    value10 = list(df3["smoker"].value_counts().keys())
    value10.sort()
    value11 = list(df4["region"].value_counts().keys())
    value11.sort()
    return render_template("index.html",value=value7, value01=value10,value02=value11)

@app.route("/predict", methods=["POST"])
def predict():
    age = request.form["age"]
    sex = request.form["sex"]
    bmi = request.form["bmi"]
    children = request.form["children"]
    smoker = request.form["smoker"]
    region = request.form["region"]
    
    predicated_price = hospital_charges_prediction_model(model, age, sex, bmi, children, smoker, region)
    predicated_price = predicated_price.astype("int")

    return render_template("index.html", prediction_text="{} clinic charges of:- {}".format(region, predicated_price))


if __name__ == "__main__":
    app.run()    
    