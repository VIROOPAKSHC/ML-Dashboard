import streamlit as st
# from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
from joblib import load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt

from lime import lime_tabular

train_df = pd.read_csv(r"kaggle-data\train.csv")
test_df = pd.read_csv(r"kaggle-data\test.csv")

X = train_df.drop("price_range",axis=1)
Y = train_df["price_range"]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

rf = load('rf_mobile.model')

Y_test_preds = rf.predict(X_test)
st.title("Mobile Price :red[Classification]")
st.markdown("Predict Mobile Price Range Class using Mobile features")

tab1,tab2,tab3 = st.tabs(["Data :clipboard:","Global Performance","Local Performance"])

with tab1:
    st.header("Mobile Price Dataset")
    st.write(train_df)

with tab2:
    col1,col2 = st.columns(2)
    with col1:
        conf_mat_fig = plt.figure(figsize=(6,6))
        ax1 = conf_mat_fig.add_subplot(111)
        skplt.metrics.plot_confusion_matrix(Y_test,Y_test_preds,ax=ax1,normalize=True)
        st.pyplot(conf_mat_fig,use_container_width=True)
    
    with col2:
        feat_imp_fig = plt.figure(figsize=(6,6))
        ax1 = feat_imp_fig.add_subplot(111)
        skplt.estimators.plot_feature_importances(rf,feature_names=train_df.columns[:-1],ax=ax1,x_tick_rotation=90)
        st.pyplot(feat_imp_fig,use_container_width=True)

    st.divider()
    st.header("Classification Report")
    st.code(classification_report(Y_test,Y_test_preds))

with tab3:
    sliders=[]
    col1,col2 = st.columns(2)
    with col1:
        for feature in train_df.columns[:-1]:
            img_sliders = st.slider(label=feature,min_value=float(train_df[feature].min()),max_value=float(train_df[feature].max()))
            sliders.append(img_sliders)
    with col2:
        prediction=rf.predict([sliders])
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("### Model Prediction : <strong style='color:tomato;'>{}</strong>".format(Y[prediction[0]]),unsafe_allow_html=True)
        probs = rf.predict_proba([sliders])

        probability = probs[0][prediction[0]]
        
        with col2:
            st.metric(label="Model Confidence",value="{:.2f} %".format(probability*100),delta="{:.2f} %".format((probability-0.5)*100))

        explainer = lime_tabular.LimeTabularExplainer(np.array(X_train),mode='classification',
                                              class_names=Y.unique(),
                                              feature_names=list(train_df.columns[:-1]))
        explanation = explainer.explain_instance(np.array(sliders),rf.predict_proba,
                                         num_features = len(list(train_df.columns[:-1])),
                                         top_labels = 3)

        interpretation_fig = explanation.as_pyplot_figure(label=prediction[0])
        st.pyplot(interpretation_fig,use_container_width=True)

