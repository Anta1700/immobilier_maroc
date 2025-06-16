import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
st.title("visualisation de ma base iimmobilier_maroc.csv")
df = pd.read_csv("immobilier_maroc.csv")
st.subheader("apercu de ma base")
st.write(df.describe())
st.subheader("ales premieres lignes")
st.write(df.head(10))



