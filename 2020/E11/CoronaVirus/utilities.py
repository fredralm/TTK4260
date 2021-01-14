#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 18:13:28 2020
https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model
@author: adil
"""
import pandas as pd
import warnings 
import matplotlib.pyplot as plt 
import numpy as np
from fbprophet.plot import add_changepoints_to_plot
from fbprophet import Prophet
from datetime import datetime 

def select_area(ncov_df, places=None, excluded_places=None):
    """
    Select the records of the palces.
    @ncov_df <pd.DataFrame>: the clean data
    @places <list[tuple(<str/None>, <str/None>)]: the list of places
        - if the list is None, all data will be used
        - (str, str): both of country and province are specified
        - (str, None): only country is specified
        - (None, str) or (None, None): Error
    @excluded_places <list[tuple(<str/None>, <str/None>)]: the list of excluded places
        - if the list is None, all data in the "places" will be used
        - (str, str): both of country and province are specified
        - (str, None): only country is specified
        - (None, str) or (None, None): Error
    @return <pd.DataFrame>: index and columns are as same as @ncov_df
    """
    # Select the target records
    df = ncov_df.copy()
    c_series = ncov_df["Country"]
    p_series = ncov_df["Province"]
    if places is not None:
        df = pd.DataFrame(columns=ncov_df.columns)
        for (c, p) in places:
            if c is None:
                raise Exception("places: Country must be specified!")
            if p is None:
                new_df = ncov_df.loc[c_series == c, :]
            else:
                new_df = ncov_df.loc[(c_series == c) & (p_series == p), :]
            df = pd.concat([df, new_df], axis=0)
    if excluded_places is not None:
        for (c, p) in excluded_places:
            if c is None:
                raise Exception("excluded_places: Country must be specified!")
            if p is None:
                df = df.loc[c_series != c, :]
            else:
                c_df = df.loc[(c_series == c) & (p_series != p), :]
                other_df = df.loc[c_series != c, :]
                df = pd.concat([c_df, other_df], axis=0)
    df = df.groupby("Date").sum().reset_index()
    return df

def show_trend(ncov_df, variable="Confirmed", n_changepoints=2, places=None, excluded_places=None):
    """
    Show trend of log10(@variable) using fbprophet package.
    @ncov_df <pd.DataFrame>: the clean data
    @variable <str>: variable name to analyse
        - if Confirmed, use Infected + Recovered + Deaths
    @n_changepoints <int>: max number of change points
    @places <list[tuple(<str/None>, <str/None>)]: the list of places
        - if the list is None, all data will be used
        - (str, str): both of country and province are specified
        - (str, None): only country is specified
        - (None, str) or (None, None): Error
    @excluded_places <list[tuple(<str/None>, <str/None>)]: the list of excluded places
        - if the list is None, all data in the "places" will be used
        - (str, str): both of country and province are specified
        - (str, None): only country is specified
        - (None, str) or (None, None): Error
    """
    # Data arrangement
    df = select_area(ncov_df, places=places, excluded_places=excluded_places)
    if variable == "Confirmed":
        df["Confirmed"] = df[["Infected", "Recovered", "Deaths"]].sum(axis=1)
    df = df.loc[:, ["Date", variable]]
    df.columns = ["ds", "y"]
    # Log10(x)
    warnings.resetwarnings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df["y"] = np.log10(df["y"]).replace([np.inf, -np.inf], 0)
    # fbprophet
    model = Prophet(growth="linear", daily_seasonality=False, n_changepoints=n_changepoints)
    model.fit(df)
    future = model.make_future_dataframe(periods=0)
    forecast = model.predict(future)
    # Create figure
    fig = model.plot(forecast)
    _ = add_changepoints_to_plot(fig.gca(), model, forecast)
    plt.title(f"log10({variable}) over time and chainge points")
    plt.ylabel(f"log10(the number of cases)")
    plt.xlabel("")
    
def create_target_df(ncov_df, total_population, places=None,
                     excluded_places=None, start_date=None, date_format="%d%b%Y"):
    """
    Select the records of the palces, calculate the number of susceptible people,
     and calculate the elapsed time [day] from the start date of the target dataframe.
    @ncov_df <pd.DataFrame>: the clean data
    @total_population <int>: total population in the places
    @places <list[tuple(<str/None>, <str/None>)]: the list of places
        - if the list is None, all data will be used
        - (str, str): both of country and province are specified
        - (str, None): only country is specified
        - (None, str) or (None, None): Error
    @excluded_places <list[tuple(<str/None>, <str/None>)]: the list of excluded places
        - if the list is None, all data in the "places" will be used
        - (str, str): both of country and province are specified
        - (str, None): only country is specified
        - (None, str) or (None, None): Error
    @start_date <str>: the start date or None
    @date_format <str>: format of @start_date
    @return <tuple(2 objects)>:
        - 1. start_date <pd.Timestamp>: the start date of the selected records
        - 2. target_df <pd.DataFrame>:
            - column T: elapsed time [min] from the start date of the dataset
            - column Susceptible: the number of patients who are in the palces but not infected/recovered/died
            - column Infected: the number of infected cases
            - column Recovered: the number of recovered cases
            - column Deaths: the number of death cases
    """
    # Select the target records
    df = select_area(ncov_df, places=places, excluded_places=excluded_places)
    if start_date is not None:
        df = df.loc[df["Date"] > datetime.strptime(start_date, date_format), :]
    start_date = df.loc[df.index[0], "Date"]
    # column T
    df["T"] = ((df["Date"] - start_date).dt.total_seconds() / 60).astype(int)
    # coluns except T
    df["Susceptible"] = total_population - df["Infected"] - df["Recovered"] - df["Deaths"]
    response_variables = ["Susceptible", "Infected", "Recovered", "Deaths"]
    # Return
    target_df = df.loc[:, ["T", *response_variables]]
    return (start_date, target_df)    