import logging

logger = logging.getLogger(__name__)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
root_logger.addHandler(sh)

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import svm
from sklearn import metrics

desired_width = 320
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', desired_width)

df = pd.read_csv("predict_stats.csv", index_col=0).reset_index(drop=True)
st.title("Player BPM/PER Prediction")

# Filter out small samples
df = df[df["mp"] > 500].reset_index(drop=True)

st.subheader("Correlations")
corr_x = st.selectbox("Correlation - X variable", options=df.columns, index=df.columns.get_loc("pts_per_g"))
corr_y = st.selectbox("Correlation - Y variable", options=["bpm", "per"], index=0)
corr_col = st.radio("Correlation - color variable", options=["age", "season"], index=1)
fig = px.scatter(df, x=corr_x, y=corr_y, title=f"Correlation {corr_x} vs {corr_y}",
                 template="plotly_white", render_mode='webgl',
                 color=corr_col, hover_data=['name', 'pos', 'age', 'season'], color_continuous_scale=px.colors.sequential.OrRd)
fig.update_traces(mode="markers", marker={"line": {"width": 0.4, "color": "slategrey"}})
st.write(fig)

# Preprocessing
# Only keep relevant features
cont_var_cols = ['g', 'mp_per_g', 'fg_per_g', 'fga_per_g', 'fg3_per_g', 'fg3a_per_g', 'fg2_per_g', 'fg2a_per_g', 'efg_pct', 'ft_per_g', 'fta_per_g','orb_per_g', 'drb_per_g', 'trb_per_g', 'ast_per_g', 'stl_per_g', 'blk_per_g', 'tov_per_g', 'pf_per_g', 'pts_per_g', 'mp']
cont_df = df[cont_var_cols]

# Scale the features
scaler = preprocessing.StandardScaler().fit(cont_df)
X = scaler.transform(cont_df)

# Split date into train/test set
st.subheader("Prediction Selection")
X_train, X_test = model_selection.train_test_split(X, train_size=0.8, random_state=42, shuffle=True)

y_stat = st.selectbox("Select Y value to predict:", ["bpm", "per"], index=0)
Y = df[y_stat].values
Y_train, Y_test = model_selection.train_test_split(Y, train_size=0.8, random_state=42, shuffle=True)

# Build models
reg_name = "Support Vector Regression"
mdl = svm.SVR(kernel='rbf', degree=3)
mdl.fit(X_train, Y_train)

# Test prediction
Y_test_hat = mdl.predict(X_test)
test_out = pd.DataFrame([Y_test_hat, Y_test], index=["Prediction", "Actual"]).transpose()
_, df_test = model_selection.train_test_split(df, train_size=0.8, random_state=42, shuffle=True)
test_out = test_out.assign(player=df_test["name"].values)
test_out = test_out.assign(season=df_test["season"].values)
val_fig = px.scatter(test_out, x="Prediction", y="Actual", title=f"{reg_name} Model: Prediction of {y_stat.upper()} vs Actual", template="plotly_white", color_discrete_sequence=px.colors.qualitative.Safe, hover_data=["player", "season"])
st.write(val_fig)

# Calculate Error
st.header("Mean Squared Error")
mse = metrics.mean_squared_error(Y_test, Y_test_hat)
st.write(f"Mean square error with {reg_name}: {round(mse, 2)}")