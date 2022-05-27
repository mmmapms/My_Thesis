from epftoolbox.evaluation import MAE, sMAPE, rMAE
from epftoolbox.data import read_data
import pandas as pd
import shap
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import numpy as np

import matplotlib.pyplot as plt



df = pd.read_csv('C:/Users/mmmap/epftoolbox/datasets/2015_BE.csv')


Y=df[' Prices (EUR_MWh)']
X=df[df.columns.difference([' Prices (EUR_MWh)', 'Date'])]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)


model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
model.fit(X_train, Y_train)  

explainer = shap.TreeExplainer(model)
shap_values=shap.TreeExplainer(model).shap_values(X_train)

def make_plots():
    shap.summary_plot(shap_values, X_train, plot_type="bar")
    f = plt.figure()
    shap.summary_plot(shap_values, X_train)
    f.savefig("C:/Users/mmmap/Desktop/epftoolbox/My_Version/summary_plot1.png", bbox_inches='tight', dpi=600)
    shap.dependence_plot("Hydro reservoir [GWh]", shap_values, X_train)

make_plots()

"""
X_output = X_test.copy()
X_output.loc[:,'predict'] = np.round(model.predict(X_output),2)

# Randomly pick some observations
neg_price=X_output['predict']<0

S = X_output[neg_price]

shap_values_Model = explainer.shap_values(S)

shap.force_plot(
    explainer.expected_value,
    shap_values_Model[3, :],
    X_output.iloc[3, :],
    matplotlib=False,
    show=False
)
plt.savefig('tmp1.svg')
plt.close()

# Get the predictions and put them with the test data.
X_output = X_test.copy()
X_output.loc[:,'predict'] = np.round(model.predict(X_output),2)

# Randomly pick some observations
neg_price=X_output['predict']<0

S = X_output[neg_price]

explainerModel = shap.TreeExplainer(model)
shap_values_Model = explainerModel.shap_values(S)

"""

