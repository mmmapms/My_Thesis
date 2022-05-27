import clock_plot as clock
import pandas as pd

Coef = pd.read_csv('C:/Users/mmmap/Desktop/Thesis/Results/LEAR_2015_BE_Coef.csv', index_col=0)


# Transforming indices to datetime format
Coef.index = pd.to_datetime(Coef.index)


# Reading data from the NP market



clock.clock_plot(Coef, "Dates", "Normalize Coef. Day-ahead generation forecast Solar (BE) (MWh)")





