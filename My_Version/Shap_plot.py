import pandas as pd
import shap

feat=["Generation forecast (FR)", "Load forecast (FR)", "Wind forecats","Solar forecast","Load forecast", "Price"]

SHAP_VALUES="C:/Users/mmmap/Desktop/Thesis/TRY_SHAPE/SHAP_VALUES.xlsx"
FEAT="C:/Users/mmmap/Desktop/Thesis/TRY_SHAPE/Features.xlsx"

Shapley = pd.read_excel(SHAP_VALUES, sheet_name=0)
Shapley = Shapley.to_numpy()
Shapley=Shapley/24

Features = pd.read_excel(FEAT, sheet_name=0)
Features = Features.to_numpy()
Features=Features/24

shap.summary_plot(Shapley, Features,feat)