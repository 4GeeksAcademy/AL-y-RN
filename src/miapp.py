import streamlit as st
from pickle import load
import sklearn

model = load(open("/workspaces/AL-y-RN/Modelo_ArbolD.pkl", "rb"))

class_dict = {"1": "si", "0": "no"}

var1 = st.slider('Numero de Embarazos', min_value=0.0, max_value=15.0, step=1.0)
var2 = st.text_input("Nivel de glucosa en sangre", value = 100)
var2 = float(var2) if var2 else 0,0
var3 = st.slider("Presion arterial", min_value=50.0, max_value=180.0, step=10.0)
var4 = st.number_input("Grosor de la piel", min_value=0.1, max_value=0.11, step=0.01)
var5 = st.text_input("Insulina")
var5 = float(var5) if var5 else 0,0
var6 = st.number_input("Indice de masa corporal")

f_opciones= {"0-4 familiares": 0.0, "5-7 familiares": 0.5, "8 o + familiares": 1}
var7 = st.selectbox("¿Cuantos familiares con diabetes tiene?", options= list(f_opciones.keys()))
var7 = f_opciones[var7]

e_opciones= {"21-35": 30, "36-50": 35, "51-65": 58, "66 o mas": 70}
var8 = st.radio("¿En que rango de edad estas?", options= list(e_opciones.keys()))
var8 = e_opciones[var8]



if st.button("predecir"):
    pred = str(model.predict([var1, var2, var3, var4, var5, var6, var7, var8])[0])
    pred_class = class_dict[pred]
    st.write(pred_class)