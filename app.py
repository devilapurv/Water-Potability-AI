import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="WaterPotability AI", page_icon="💧", layout="wide")
FEATURES = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]

st.markdown("""<style>
.stApp {background:#071321;color:#e9f8ff}
.block-container {max-width:1250px;padding-top:1.5rem}
h1,h2,h3 {color:#ecfbff!important}
.hero {padding:85px 25px;border-radius:24px;text-align:center;background:radial-gradient(ellipse at 50% 15%,#12627a 0%,#102a40 38%,#071321 85%);border:1px solid #276075;margin:20px 0 34px}
.hero h1 {font-size:clamp(38px,6vw,70px);line-height:1.1;font-weight:850}
.hero p {font-size:18px;color:#b3cddb;max-width:740px;margin:18px auto}
.aqua {color:#46e0eb}
.feature {padding:26px;border:1px solid #27475e;border-radius:18px;background:#11283b;min-height:180px}
.feature h3 {color:#49e0eb!important}
.feature p {color:#b3c8d6}
[data-testid="stMetric"] {background:#11283b;border:1px solid #28495d;border-radius:14px;padding:16px}
.stButton>button[kind="primary"],.stDownloadButton>button {background:#17c8d9;color:#06131d;border:0;border-radius:9px;font-weight:700}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    return joblib.load("water_model.pkl"), joblib.load("imputer.pkl")

st.markdown("### 💧 WaterPotability AI")
page = st.radio("Navigate", ["Home", "Single Prediction", "Bulk Analysis", "Model Insights"], horizontal=True, label_visibility="collapsed")
st.divider()

if page == "Home":
    st.markdown("""<div class="hero"><h1>Understand Your Water<br><span class="aqua">With Artificial Intelligence</span></h1><p>Explore water quality measurements, discover patterns, and generate experimental machine learning predictions.</p><p>💧 Machine Learning &nbsp; • &nbsp; 📊 Analytics &nbsp; • &nbsp; ⚡ Instant Predictions</p></div>""", unsafe_allow_html=True)
    cols=st.columns(3)
    cards=[("💧 Single Prediction","Enter nine water-quality measurements and analyze a sample."),("📂 Bulk Analysis","Upload unseen CSV samples and download model predictions."),("📊 Model Insights","Explore model performance and feature importance.")]
    for c,(title,description) in zip(cols,cards):
        with c: st.markdown(f'<div class="feature"><h3>{title}</h3><p>{description}</p></div>',unsafe_allow_html=True)
    st.markdown("## About this project")
    st.write("A Random Forest classification demonstration built with scikit-learn and Streamlit. The dataset's provenance is uncertain; predictions are not drinking-water safety certifications.")
else:
    if not Path("water_model.pkl").exists() or not Path("imputer.pkl").exists():
        st.error("Missing model files. Add your own water_model.pkl and imputer.pkl from Google Colab to this folder, then reload the app.")
        st.stop()
    try:
        model,imputer=load_assets()
    except Exception as e:
        st.error(f"Could not load saved model files: {e}")
        st.stop()
    def predict(frame):
        clean = frame[FEATURES].fillna(
    dict(zip(FEATURES, imputer.statistics_))
)
        labels=model.predict(clean)
        probability=model.predict_proba(clean)[:,list(model.classes_).index(1)]
        return labels,probability
    if page == "Single Prediction":
        st.title("💧 Single Water Sample")
        st.write("Enter the nine water-quality measurements.")
        defaults=[7.0,200.0,20000.0,7.0,330.0,420.0,14.0,65.0,4.0]
        with st.form("single"):
            columns=st.columns(3)
            vals=[]
            for i,(name,default) in enumerate(zip(FEATURES,defaults)):
                with columns[i%3]:
                    vals.append(st.number_input(name,min_value=0.0,max_value=14.0 if name=="ph" else None,value=default))
            submitted=st.form_submit_button("Analyze Sample",type="primary",use_container_width=True)
        if submitted:
            sample=pd.DataFrame([vals],columns=FEATURES)
            labels,prob=predict(sample)
            st.subheader("Analysis Result")
            st.info("Predicted class: **"+("Potable (1)" if labels[0]==1 else "Non-Potable (0)")+"**")
            st.metric("Model probability for class 1",f"{prob[0]:.1%}")
            st.progress(float(prob[0]))
            st.warning("Not a laboratory water safety test. Do not rely on this prediction to decide whether to drink water.")
    elif page == "Bulk Analysis":
        st.title("📂 Bulk CSV Analysis")
        st.write("Upload unseen samples with all nine required feature columns. Extra columns are preserved in the output.")
        template=pd.DataFrame([[7,200,20000,7,330,420,14,65,4]],columns=FEATURES)
        st.download_button("Download CSV Template",template.to_csv(index=False),"water_template.csv","text/csv")
        upload=st.file_uploader("Upload CSV",type="csv")
        if upload:
            try:
                data=pd.read_csv(upload)
                st.dataframe(data.head(),use_container_width=True)
                missing=[x for x in FEATURES if x not in data.columns]
                if missing: st.error("Missing columns: "+", ".join(missing))
                elif data.empty: st.error("The CSV has no rows.")
                else:
                    raw=data[FEATURES].copy()
                    numeric=raw.apply(pd.to_numeric,errors="coerce")
                    invalid=((raw.notna()) & numeric.isna()).any(axis=1)
                    if invalid.any(): st.error(f"{int(invalid.sum())} rows contain nonnumeric text. Correct the CSV and try again.")
                    elif numeric.isna().all(axis=0).any(): st.error("At least one feature column has no usable numeric values.")
                    elif (numeric < 0).any().any(): st.error("Measurements cannot be negative.")
                    elif ((numeric.ph < 0)|(numeric.ph > 14)).any(): st.error("pH must be between 0 and 14.")
                    else:
                        labels,probs=predict(numeric)
                        result=data.copy()
                        result["Prediction"]=labels
                        result["Predicted_Class"]=np.where(labels==1,"Potable","Non-Potable")
                        result["Potable_Probability"]=probs
                        st.success("Predictions generated!")
                        a,b,c=st.columns(3)
                        a.metric("Total samples",len(result));b.metric("Predicted potable",int((labels==1).sum()));c.metric("Predicted non-potable",int((labels==0).sum()))
                        chart=result.Predicted_Class.value_counts().rename_axis("Class").reset_index(name="Samples")
                        fig=px.pie(chart,names="Class",values="Samples",hole=.55,color="Class",color_discrete_map={"Potable":"#20d6dd","Non-Potable":"#ff7487"},template="plotly_dark")
                        fig.update_layout(paper_bgcolor="#071321")
                        st.plotly_chart(fig,use_container_width=True)
                        st.dataframe(result,use_container_width=True)
                        st.download_button("Download Predictions",result.to_csv(index=False),"water_predictions.csv","text/csv")
                        st.warning("Predictions are experimental and do not certify drinking-water safety.")
            except Exception as e: st.error(f"Unable to process file: {e}")
    else:
        st.title("📊 Model Insights")
        a,b,c=st.columns(3)
        a.metric("Test accuracy","67.07%")
        b.metric("Non-potable recall","91%")
        c.metric("Potable recall","30%")
        st.caption("Metrics reported from the original held-out test set (656 samples).")
        importance=pd.DataFrame({"Feature":FEATURES,"Importance":model.feature_importances_}).sort_values("Importance")
        fig=px.bar(importance,x="Importance",y="Feature",orientation="h",template="plotly_dark",color_discrete_sequence=["#26d8e4"])
        fig.update_layout(paper_bgcolor="#071321")
        st.plotly_chart(fig,use_container_width=True)
        st.warning("Model accuracy is modest, and class-1 recall is low. Model probabilities are not calibrated safety scores. The dataset's real-world provenance is uncertain.")
st.divider()
st.caption("💧 WaterPotability AI • Python • scikit-learn • Streamlit • Educational demonstration")
