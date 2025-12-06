# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io
from datetime import datetime
from pathlib import Path

# Streamlit page config
st.set_page_config(page_title="Traffic Accident Severity Dashboard", layout="wide")

# ========== Styling ==========
st.markdown(
    """
    <style>
    .reportview-container {background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);}
    .css-1aumxhk {background-color: transparent;}
    h1,h2,h3 {color:#FFB86B; text-shadow: 1px 1px 6px rgba(0,0,0,0.6);}
    .stButton>button {background-color:#FF6B6B;color:white;border-radius:8px;}
    .stDownloadButton>button {background-color:#4CAF50;color:white;border-radius:8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ========== Helpers ==========
@st.cache_data
def load_data(path: str):
    df_local = pd.read_csv(path)
    return df_local

def safe_hour_extract(df, time_col="Time"):
    if time_col in df.columns:
        # try to parse times like hh:mm or hh:mm:ss, fallback fillna 12
        df["Hour"] = pd.to_datetime(df[time_col], errors="coerce").dt.hour.fillna(12).astype(int)
    else:
        df["Hour"] = 12
    return df

def prepare_model(path="accident_model.pkl"):
    """Try to load joblib model. Return (model, meta_dict) or (None, None)."""
    if not Path(path).exists():
        return None, None
    try:
        obj = joblib.load(path)
        # support different save formats
        if isinstance(obj, tuple):
            if len(obj) == 2:
                model, cat_cols = obj
                meta = {"cat_cols": cat_cols}
                return model, meta
            elif len(obj) == 3:
                model, cat_cols, num_cols = obj
                meta = {"cat_cols": cat_cols, "num_cols": num_cols}
                return model, meta
        else:
            # single object saved (model)
            model = obj
            return model, {}
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    buf.seek(0)
    return buf

# ========== Load dataset ==========
DATA_PATH = "RTA_Dataset.csv"
if not Path(DATA_PATH).exists():
    st.error(f"Dataset file '{DATA_PATH}' not found. Please place it in the app folder.")
    st.stop()

df = load_data(DATA_PATH)

# Basic cleaning: unify common missing markers and strip spaces
df = df.replace(["?", "NA", "N/A", "na", "--", ""], np.nan)
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# derive hour
df = safe_hour_extract(df, "Time")

# Ensure expected TARGET exists
TARGET = "Accident_severity"
if TARGET not in df.columns:
    st.error(f"Target column '{TARGET}' not present in dataset.")
    st.stop()

# Identify columns sets
all_columns = df.columns.tolist()
cat_columns = df.select_dtypes(include="object").columns.tolist()
num_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Prepare categories, fill missing for display / model
for c in cat_columns:
    df[c] = df[c].astype(str).fillna("Unknown")

for n in num_columns:
    df[n] = pd.to_numeric(df[n], errors="coerce").fillna(0)

# ========== Load model ==========
model, meta = prepare_model("accident_model.pkl")
if model is None:
    st.warning("No saved model found (accident_model.pkl). To enable prediction tab, run training script first.")
else:
    st.success("Loaded saved model for predictions.")

# class names fallback
if model is not None:
    try:
        class_names = list(model.classes_)
    except:
        class_names = sorted(df[TARGET].unique().tolist())
else:
    class_names = sorted(df[TARGET].unique().tolist())

# ========== Sidebar filters ==========
st.sidebar.title("Filters & Controls")
st.sidebar.markdown("Narrow the dataset for exploration")

# Weather, Day, Vehicle etc (use safe unique lists)
weather_opts = sorted(df["Weather_conditions"].unique().tolist()) if "Weather_conditions" in df.columns else []
day_opts = sorted(df["Day_of_week"].unique().tolist()) if "Day_of_week" in df.columns else []
vehicle_opts = sorted(df["Type_of_vehicle"].unique().tolist()) if "Type_of_vehicle" in df.columns else []
area_opts = sorted(df["Area_accident_occured"].unique().tolist()) if "Area_accident_occured" in df.columns else []

sel_weather = st.sidebar.multiselect("Weather", options=weather_opts, default=weather_opts[:3])
sel_day = st.sidebar.multiselect("Day of Week", options=day_opts, default=day_opts[:7])
sel_vehicle = st.sidebar.multiselect("Vehicle Type", options=vehicle_opts, default=vehicle_opts[:5])
sel_area = st.sidebar.multiselect("Area", options=area_opts, default=[])

hour_min, hour_max = st.sidebar.slider("Hour range", 0, 23, (0, 23))
casualty_range = st.sidebar.slider("Number of casualties", 0, int(df["Number_of_casualties"].max()), (0, 5))

# Apply filters
filtered = df.copy()
if sel_weather:
    filtered = filtered[filtered["Weather_conditions"].isin(sel_weather)]
if sel_day:
    filtered = filtered[filtered["Day_of_week"].isin(sel_day)]
if sel_vehicle:
    filtered = filtered[filtered["Type_of_vehicle"].isin(sel_vehicle)]
if sel_area:
    filtered = filtered[filtered["Area_accident_occured"].isin(sel_area)]

filtered = filtered[(filtered["Hour"].between(hour_min, hour_max)) &
                    (filtered["Number_of_casualties"].between(casualty_range[0], casualty_range[1]))]

# ========== Top metrics ==========
st.title("🚦 Traffic Accident Severity Dashboard")
st.markdown("Interactive EDA and prediction tool for road traffic accidents.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Accidents", len(filtered))
c2.metric("Fatal (count)", int((filtered[TARGET] == "Fatal injury").sum()))
c3.metric("Serious (count)", int((filtered[TARGET] == "Serious Injury").sum()))
c4.metric("Slight (count)", int((filtered[TARGET] == "Slight Injury").sum()))

# ========== Tabs ==========
tab_overview, tab_env, tab_human, tab_prediction, tab_model = st.tabs(
    ["Overview", "Environment & Road", "Human & Vehicle", "Predict", "Model & Insights"]
)

# ------------------ Overview Tab ------------------
with tab_overview:
    st.header("Overview")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Severity Breakdown")
        fig, ax = plt.subplots(figsize=(6,4))
        counts = filtered[TARGET].value_counts()
        colors = ["#FF6B6B", "#FFB86B", "#8BE78B"][:len(counts)]
        counts.plot(kind="pie", autopct="%1.1f%%", startangle=90, ax=ax, colors=colors)
        ax.set_ylabel("")
        st.pyplot(fig)

    with col2:
        st.subheader("Top 10 Accident Areas")
        if "Area_accident_occured" in filtered.columns:
            top_area = filtered["Area_accident_occured"].value_counts().nlargest(10)
            fig2, ax2 = plt.subplots(figsize=(6,4))
            sns.barplot(x=top_area.values, y=top_area.index, palette="magma", ax=ax2)
            ax2.set_xlabel("Count")
            st.pyplot(fig2)
        else:
            st.info("No area data available.")

    st.markdown("### Accidents by Hour (filtered)")
    fig_hour, ax_hour = plt.subplots(figsize=(10,3))
    filtered.groupby("Hour").size().reindex(range(24), fill_value=0).plot(kind="bar", ax=ax_hour)
    ax_hour.set_xlabel("Hour of day")
    ax_hour.set_ylabel("Accident count")
    st.pyplot(fig_hour)

    st.markdown("### Download filtered data")
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV of filtered data", data=csv_bytes, file_name="filtered_accidents.csv", mime="text/csv")

# ------------------ Environment & Road Tab ------------------
with tab_env:
    st.header("Environment & Road Analysis")
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Severity by Weather")
        if "Weather_conditions" in filtered.columns:
            fig_w, ax_w = plt.subplots(figsize=(8,4))
            sns.countplot(x="Weather_conditions", hue=TARGET, data=filtered, palette="rocket", ax=ax_w)
            ax_w.set_xticklabels(ax_w.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig_w)
        else:
            st.info("Weather data not present.")

        st.subheader("Severity by Light Conditions")
        if "Light_conditions" in filtered.columns:
            fig_l, ax_l = plt.subplots(figsize=(8,4))
            sns.countplot(x="Light_conditions", hue=TARGET, data=filtered, palette="viridis", ax=ax_l)
            ax_l.set_xticklabels(ax_l.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig_l)

    with cols[1]:
        st.subheader("Road Surface Type & Condition")
        if "Road_surface_type" in filtered.columns:
            fig_r, ax_r = plt.subplots(figsize=(8,4))
            sns.countplot(x="Road_surface_type", hue=TARGET, data=filtered, palette="cubehelix", ax=ax_r)
            ax_r.set_xticklabels(ax_r.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig_r)

        if "Road_surface_conditions" in filtered.columns:
            fig_rc, ax_rc = plt.subplots(figsize=(8,4))
            sns.countplot(x="Road_surface_conditions", hue=TARGET, data=filtered, palette="mako", ax=ax_rc)
            ax_rc.set_xticklabels(ax_rc.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig_rc)

# ------------------ Human & Vehicle Tab ------------------
with tab_human:
    st.header("Human & Vehicle Factors")
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Driver Age Band vs Severity")
        if "Age_band_of_driver" in filtered.columns:
            fig_da, ax_da = plt.subplots(figsize=(8,4))
            sns.countplot(x="Age_band_of_driver", hue=TARGET, data=filtered, palette="flare", ax=ax_da)
            ax_da.set_xticklabels(ax_da.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig_da)

        st.subheader("Driving Experience vs Severity")
        if "Driving_experience" in filtered.columns:
            fig_de, ax_de = plt.subplots(figsize=(8,4))
            sns.countplot(x="Driving_experience", hue=TARGET, data=filtered, palette="crest", ax=ax_de)
            ax_de.set_xticklabels(ax_de.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig_de)

    with colB:
        st.subheader("Vehicle Type vs Severity")
        if "Type_of_vehicle" in filtered.columns:
            fig_v, ax_v = plt.subplots(figsize=(8,4))
            sns.countplot(x="Type_of_vehicle", hue=TARGET, data=filtered, palette="cividis", ax=ax_v)
            ax_v.set_xticklabels(ax_v.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig_v)

        st.subheader("Vehicle Defects (Top)")
        if "Defect_of_vehicle" in filtered.columns:
            top_def = filtered["Defect_of_vehicle"].value_counts().nlargest(10)
            fig_def, ax_def = plt.subplots(figsize=(8,4))
            sns.barplot(x=top_def.values, y=top_def.index, palette="rocket", ax=ax_def)
            st.pyplot(fig_def)

# ------------------ Predict Tab ------------------
with tab_prediction:
    st.header("Predict Accident Severity (interactive)")
    if model is None:
        st.warning("No trained model available. Please run train_model.py to create 'accident_model.pkl' then reload app.")
    # Build input UI using model.feature_names_ when possible
    # Fallback: show common inputs
    feat_list = None
    if model is not None and hasattr(model, "feature_names_"):
        feat_list = list(model.feature_names_)
    else:
        # use a sensible default set of features from dataset
        feat_list = ["Day_of_week", "Driving_experience", "Weather_conditions",
                     "Road_surface_conditions", "Number_of_vehicles_involved",
                     "Number_of_casualties", "Hour"]

    st.markdown("Provide input values (any fields left blank will use sensible defaults).")
    col1, col2, col3 = st.columns(3)

    # create inputs with dataset uniques if available
    day_inp = col1.selectbox("Day of Week", options=df["Day_of_week"].unique().tolist())
    exp_inp = col1.selectbox("Driving Experience", options=df["Driving_experience"].unique().tolist() if "Driving_experience" in df.columns else ["Unknown"])
    weather_inp = col2.selectbox("Weather", options=df["Weather_conditions"].unique().tolist() if "Weather_conditions" in df.columns else ["Unknown"])
    road_inp = col2.selectbox("Road Surface Condition", options=df["Road_surface_conditions"].unique().tolist() if "Road_surface_conditions" in df.columns else ["Unknown"])
    nv_inp = col3.number_input("Number of Vehicles", min_value=0, max_value=50, value=1)
    nc_inp = col3.number_input("Number of Casualties", min_value=0, max_value=50, value=1)
    hour_inp = col3.slider("Hour (0-23)", 0, 23, 12)

    if st.button("🔍 Predict"):
        # Build single-row input matching model.feature_names_
        input_row = {}
        for f in feat_list:
            if f == "Day_of_week":
                input_row[f] = day_inp
            elif f == "Driving_experience":
                input_row[f] = exp_inp
            elif f == "Weather_conditions":
                input_row[f] = weather_inp
            elif f == "Road_surface_conditions":
                input_row[f] = road_inp
            elif f == "Number_of_vehicles_involved":
                input_row[f] = int(nv_inp)
            elif f == "Number_of_casualties":
                input_row[f] = int(nc_inp)
            elif f == "Hour":
                input_row[f] = int(hour_inp)
            else:
                # default fallback - if feature exists in dataset, try using a sensible default
                if f in df.columns:
                    # take the most common value
                    input_row[f] = df[f].mode().iloc[0]
                else:
                    input_row[f] = "Unknown"

        X_input = pd.DataFrame([input_row], columns=feat_list)

        # Ensure categorical fields are strings
        for c in X_input.select_dtypes(include="object").columns:
            X_input[c] = X_input[c].astype(str)

        # Make prediction
        try:
            pred = model.predict(X_input)[0]
            try:
                pred_label = class_names[int(pred)]
            except:
                pred_label = str(pred)
            st.success(f"Predicted Accident Severity: **{pred_label}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            pred_label = None

        # Show local feature contributions (CatBoost prediction values change)
        if pred_label is not None:
            with st.expander("🔎 Show feature contributions for this prediction"):
                try:
                    from catboost import Pool
                    # if meta provides cat_cols, convert indices
                    cat_cols_meta = meta.get("cat_cols", [])
                    # Build pool
                    pool = Pool(X_input, cat_features=[X_input.columns.get_loc(c) for c in cat_cols_meta if c in X_input.columns])
                    contrib = model.get_feature_importance(type="PredictionValuesChange", data=pool)[0]
                    contrib_df = pd.DataFrame({"feature": feat_list, "impact": contrib})
                    contrib_df = contrib_df.reindex(contrib_df["impact"].abs().sort_values(ascending=False).index)
                    fig_c, ax_c = plt.subplots(figsize=(8,5))
                    sns.barplot(x="impact", y="feature", data=contrib_df.head(12), palette="coolwarm", ax=ax_c)
                    plt.title("Top feature impacts for this prediction")
                    st.pyplot(fig_c)
                    st.table(contrib_df.head(12).assign(impact=lambda d: d["impact"].round(4)))
                except Exception as e:
                    st.warning(f"Could not compute local contributions: {e}")

# ------------------ Model & Insights Tab ------------------
with tab_model:
    st.header("Model & Global Insights")
    if model is None:
        st.info("No trained model found. Run `train_model.py` to produce 'accident_model.pkl'.")
    else:
        # Global feature importance
        try:
            st.subheader("Global Feature Importance")
            importances = model.get_feature_importance()
            feat_names = list(model.feature_names_)
            fi_df = pd.Series(importances, index=feat_names).sort_values(ascending=True).tail(20)
            fig_fi, ax_fi = plt.subplots(figsize=(8,6))
            fi_df.plot(kind="barh", ax=ax_fi, color="teal")
            st.pyplot(fig_fi)
            st.write("Top features:", list(fi_df.tail(10).index[::-1]))
        except Exception as e:
            st.warning(f"Could not compute feature importance: {e}")

        # Model performance on training data (quick snapshot)
        st.subheader("Model Performance (train dataset snapshot)")
        try:
            # If dataset columns used to train exist, predict on entire df
            X_full = df.drop(columns=[TARGET])
            # Ensure columns match model.feature_names_ before predict
            if hasattr(model, "feature_names_"):
                cols_req = list(model.feature_names_)
                # build X_pred consistent with model.features
                X_pred = pd.DataFrame(columns=cols_req)
                for c in cols_req:
                    if c in X_full.columns:
                        X_pred[c] = X_full[c]
                    else:
                        # add default value
                        X_pred[c] = [df[c].mode().iloc[0]] * len(df) if c in df.columns else ["Unknown"] * len(df)
                y_pred = model.predict(X_pred)
                from sklearn.metrics import classification_report, confusion_matrix
                y_true = df[TARGET].astype(str)
                # try decoding if model returns numeric labels
                try:
                    y_pred_labels = [class_names[int(x)] for x in y_pred]
                except:
                    y_pred_labels = [str(x) for x in y_pred]
                report = classification_report(y_true, y_pred_labels, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose().round(3)
                st.dataframe(report_df)
                st.write("Confusion matrix (train):")
                cm = confusion_matrix(y_true, y_pred_labels, labels=report_df.index[:-3])
                st.write(cm)
        except Exception as e:
            st.warning(f"Could not compute model metrics on dataset: {e}")

st.markdown("------")
st.markdown("Built with ❤️")
