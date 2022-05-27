
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot

# Download available forecast of the NP market available in the library repository
# These forecasts accompany the original paper
forecast = pd.read_csv('C:/Users/mmmap/Desktop/Thesis/Results/LEAR_2015_NP_Coef.csv', index_col=0)

# Transforming indices to datetime format
forecast.index = pd.to_datetime(forecast.index)



# Extracting forecast of DNN ensemble and display
MAE_hour = forecast.loc[:, ['MAE']]

# Extracting real price and display
plot_acf(MAE_hour, lags=90)
pyplot.show()





