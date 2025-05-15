import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import pickle
import sklearn

st.title("Análisis promedio académicos")

tab1, tab2, tab3 = st.tabs(["Análisis univariado","Análisis bivariado", "Arbol de decisión"])

with open("model.pickle", "rb") as f:
    modelo = pickle.load(f)

prom=pd.read_csv("prom.csv")


with tab1:
    fig, ax = plt.subplots(1,1)
    ax.hist(prom["promedio"].dropna())
    ax.set_title("Promedio")
    ax.axis()
    st.pyplot(fig)

    fig, ax = plt.subplots(1,1)
    ax.hist(prom["edad"].dropna())
    ax.set_title("Edad")
    ax.axis()
    st.pyplot(fig)

    fig, ax = plt.subplots(1,1)
    ax.hist(prom["computador"].dropna())
    ax.set_title("Computador")
    ax.axis()
    st.pyplot(fig)

    fig, ax = plt.subplots(1,1)
    ax.hist(prom["inasistencias"].dropna())
    ax.set_title("Inasistencias")
    ax.axis()
    st.pyplot(fig)




with tab2:
    fig, ax = plt.subplots(1,1)
    sns.scatterplot(y = prom["promedio"], x = prom["inasistencias"].dropna())
    ax.set_title("Promedio vs. Inasistencias")
    ax.axis()
    st.pyplot(fig)

    fig, ax = plt.subplots(1,1)
    sns.scatterplot(y = prom["promedio"], x = prom["edad"].dropna())
    ax.set_title("Promedio vs. Edad")
    ax.axis()
    st.pyplot(fig)

    fig, ax = plt.subplots(1,1)
    sns.boxplot(y = prom["promedio"], x = prom["computador"].dropna())
    ax.set_title("Promedio vs. IComputador")
    ax.axis()
    st.pyplot(fig)






with tab3:
    edad = st.slider("Edad", 0.0, 60.0)
    inasistencia = st.slider("Inasistencia", 0.0, 10.0)
    computador = st.selectbox("Computador", ["Si", "No"])
    if computador == "Si":
        computador = 1
    else:
        computador = 0
    if st.button("Predecir"):
        pred = modelo.predict(np.array([[edad, inasistencia, computador]]))
        st.write(f"Su promedio sería {round(pred[0], 1)}")

