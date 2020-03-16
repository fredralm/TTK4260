#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 15:24:00 2020
https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model
@author: adil
"""

import pandas as pd
from plot import line_plot
import matplotlib.pyplot as plt
from utilities import select_area, show_trend, create_target_df
import seaborn as sns
from datetime import datetime 
from models import SIR,SIRD

time_format = "%d%b%Y %H:%M"
datetime.now().strftime(time_format)

raw = pd.read_csv("./data/covid_19_data.csv")
data_cols = ["Infected", "Deaths", "Recovered"]
rate_cols = ["Fatal per Confirmed", "Recovered per Confirmed", "Fatal per (Fatal or Recovered)"]
variable_dict = {"Susceptible": "S", "Infected": "I", "Recovered": "R", "Deaths": "D"}
ncov_df = raw.rename({"ObservationDate": "Date", "Province/State": "Province"}, axis=1)
ncov_df["Date"] = pd.to_datetime(ncov_df["Date"])
ncov_df["Country"] = ncov_df["Country/Region"].replace(
    {
        "Mainland China": "China",
        "Hong Kong SAR": "Hong Kong",
        "Taipei and environs": "Taiwan",
        "Iran (Islamic Republic of)": "Iran",
        "Republic of Korea": "South Korea",
        "Republic of Ireland": "Ireland",
        "Macao SAR": "Macau",
        "Russian Federation": "Russia",
        "Republic of Moldova": "Moldova",
        "Taiwan*": "Taiwan",
        "Cruise Ship": "Others",
        "United Kingdom": "UK",
        "Viet Nam": "Vietnam",
        "Czechia": "Czech Republic",
        "St. Martin": "Saint Martin",
        "Cote d'Ivoire": "Ivory Coast",
        "('St. Martin',)": "Saint Martin",
        "Congo (Kinshasa)": "Congo",
    }
)
ncov_df["Province"] = ncov_df["Province"].fillna("-").replace(
    {
        "Cruise Ship": "Diamond Princess cruise ship",
        "Diamond Princess": "Diamond Princess cruise ship"
    }
)

ncov_df["Infected"] = ncov_df["Confirmed"] - ncov_df["Deaths"] - ncov_df["Recovered"]
ncov_df[data_cols] = ncov_df[data_cols].astype(int)
ncov_df = ncov_df.loc[:, ["Date", "Country", "Province", *data_cols]]
print(ncov_df.tail())
print(ncov_df.info())
ncov_df.describe(include="all").fillna("-")
pd.DataFrame(ncov_df.isnull().sum()).T
", ".join(ncov_df["Country"].unique().tolist())


total_df = ncov_df.loc[ncov_df["Country"] != "China", :].groupby("Date").sum()
total_df[rate_cols[0]] = total_df["Deaths"] / total_df[data_cols].sum(axis=1)
total_df[rate_cols[1]] = total_df["Recovered"] / total_df[data_cols].sum(axis=1)
total_df[rate_cols[2]] = total_df["Deaths"] / (total_df["Deaths"] + total_df["Recovered"])
total_df.tail()
line_plot(total_df[data_cols], title="Cases over time (Total except China)")
line_plot(total_df[rate_cols], "Rate over time (Total except China)", ylabel="", math_scale=False)
total_df[rate_cols].plot.kde()
plt.title("Kernel density estimation of the rates (Total except China)")
plt.show()


population_date = "15Mar2020"
_dict = {
    "Global": "7,794,798,729",
    "China": "1,439,323,774",
    "Japan": "126,476,458",
    "South Korea": "51,269,182",
    "Italy": "60,461,827",
    "Iran": "83,992,953",
    "Norway":"5,421,241",
    "India": "1,380,004,385",
    "USA": "331,002,647",

}
population_dict = {k: int(v.replace(",", "")) for (k, v) in _dict.items()}
df = pd.io.json.json_normalize(population_dict)
df.index = [f"Total population on {population_date}"]

train_start_date, train_df = create_target_df(
    ncov_df, population_dict["Global"] - population_dict["China"], excluded_places=[("China", None)]
)
train_start_date.strftime(time_format)

#To get the timeseries for any particular coutry
#train_start_date, train_df =create_target_df(ncov_df, population_dict["India"], places=[('India',None)])



# Feature engineering for SIR model
df = train_df.rename(variable_dict, axis=1)
for (_, v) in variable_dict.items():
    df[f"d{v}/dT"] = df[v].diff() / df["T"].diff()
    if v == "S":
        N = population_dict["Global"] - population_dict["China"]
        df["-(dS/dT)*(N/S)"] = 0 - df["S"].diff() / df["T"].diff() / df["S"] * N
df.set_index("T").corr().loc[variable_dict.values(), :].style.background_gradient(axis=None)

print(df[["I", "-(dS/dT)*(N/S)", "dI/dT", "dR/dT", "dD/dT"]].tail())
sns.lmplot(
    x="I", y="value", col="diff", sharex=False, sharey=False,
    data=df[["I", "-(dS/dT)*(N/S)", "dI/dT", "dR/dT", "dD/dT"]].melt(id_vars="I", var_name="diff")
)
plt.show()

train_dataset = SIR.create_dataset(
    ncov_df, population_dict["Global"] - population_dict["China"], excluded_places=[("China", None)]
)
train_start_date, train_initials, train_Tend, train_df = train_dataset
print([train_start_date.strftime(time_format), train_initials, train_Tend])

print(train_df.tail())

line_plot(
    train_df.set_index("T").drop("x", axis=1),
    "Training data: y(T), z(T)", math_scale=False, ylabel=""
)








