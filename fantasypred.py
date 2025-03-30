import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st
import matplotlib.pyplot as plt  
import plotly.express as px


def player_data(player_name):
    first = player_name.split(" ")[0][0:2]
    first = re.sub(r'[^a-zA-Z]', '', first)
    last = " "
    if len(player_name.split(" ")[-1]) >=4:
        last = player_name.split(" ")[-1][0:4]
    else:
        last = player_name.split(" ")[-1][0:3] + "x"
    last = re.sub(r'[^a-zA-Z]', '', last)
    try:
        link = "https://www.pro-football-reference.com/players/" + last[0] + "/" + last[0:4] + first[0:2] + "00/gamelog/"
        st.write(link)
        data = pd.read_html(link)[0]
        data.columns = [" ".join(col).strip() for col in data.columns.values]
        data["Year"] = data["Unnamed: 4_level_0 Date"].str.split('-').str[0]
        data = data[data.get("Year") != "Date"].drop(data.shape[0] - 1)
        convert = ["Year", "Unnamed: 2_level_0 Gtm", "Unnamed: 3_level_0 Week", "Receiving Tgt", "Receiving Rec", "Receiving Yds", "Receiving TD", "Passing Yds", "Passing TD", "Passing Int", "Fumbles Fmb", "Rushing Yds", "Rushing TD"]
        for column in convert:
            try:
                data[column] = pd.to_numeric(data[column])
            except:
                continue
        data_years = data[data.get("Year") >= 2023]
        if max(data.get("Year")) < 2023:
            raise Exception()
        data_years = data_years.reset_index(drop = True)
        data_years["Fantasy"] = data_years["Receiving Yds"] * 0.1 + data_years["Rushing Yds"] * 0.1 + data_years["Receiving TD"] * 6 + data_years["Rushing TD"] * 6 + data_years["Receiving Rec"] * 1 +  data_years["Passing Yds"] * 0.04 +  data_years["Passing TD"] * 4 +  data_years["Passing Int"] * -2 +  data_years["Fumbles Fmb"] * -2
        data_years["Name"] = player_name
    except:
        print("Except needed")
        data = pd.read_html("https://www.pro-football-reference.com/players/" + last[0] + "/" + last[0:4] + first[0:2] + "01/gamelog/")[0]
        data.columns = [" ".join(col).strip() for col in data.columns.values]
        data["Year"] = data["Unnamed: 4_level_0 Date"].str.split('-').str[0]
        data = data[data.get("Year") != "Date"].drop(data.shape[0] - 1)
        convert = ["Year", "Unnamed: 2_level_0 Gtm", "Unnamed: 3_level_0 Week", "Receiving Tgt", "Receiving Rec", "Receiving Yds", "Receiving TD", "Passing Yds", "Passing TD", "Passing Int", "Fumbles Fmb", "Rushing Yds", "Rushing TD"]
        for column in convert:
            try:
                data[column] = pd.to_numeric(data[column])
            except:
                continue
        data_years = data[data.get("Year") >= 2023] #except
        data_years = data_years.reset_index(drop = True)
        try:
            data_years["Fantasy"] = data_years["Receiving Yds"] * 0.1 + data_years["Rushing Yds"] * 0.1 + data_years["Receiving TD"] * 6 + data_years["Rushing TD"] * 6 + data_years["Receiving Rec"] * 1 +  data_years["Passing Yds"] * 0.04 +  data_years["Passing TD"] * 4 +  data_years["Passing Int"] * -2 +  data_years["Fumbles Fmb"] * -2
        except:
            data_years["Fantasy"] = data_years["Rushing Yds"] * 0.1 + data_years["Rushing TD"] * 6 +  data_years["Passing Yds"] * 0.04 +  data_years["Passing TD"] * 4 +  data_years["Passing Int"] * -2 +  data_years["Fumbles Fmb"] * -2

        data_years["Name"] = player_name
    return data_years

def prediction(data):
    X = data.drop("Fantasy", axis = 1)
    y = data["Fantasy"]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    pipe = Pipeline([
        ("ohe", OneHotEncoder()),
        ("poly", PolynomialFeatures(degree=3)),
        ("linear", LinearRegression())
    ])
    pipe.fit(X_test, y_test)
    y_pred = pipe.predict(X_test)
    return X, y, y_pred

st.title("Fantasy Football Predictions")
player_name = st.text_input("Enter a player name: ")
if player_name:
    data = player_data(player_name)
    st.dataframe(data)
    preds = prediction(data)
    overall = preds[2].mean()
    st.metric(label="Overall Predicted Fantasy Score", value=overall.round(2))
    plotx = preds[0].index
    ploty = preds[1]
    fig = px.line(x = plotx, y = ploty, labels=dict(x="Games", y="Fantasy Points"), title = "Fantasy Points Per Game")
    fig.add_hline(y=overall, line_color = "red")
    st.plotly_chart(fig)
else:
    st.warning("Input a player name!")
