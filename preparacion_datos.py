import streamlit as st
import pandas as pd

@st.cache_data
def cargar_datos(archivo):
    if archivo.name.endswith('csv'):
        df = pd.read_csv(archivo)
    elif archivo.name.endswith('xls'):
        df = pd.read_excel(archivo)
    else:
        raise Exception('Este formato no es compatible')
    return df

def mostrar_cargar_datos():
    st.subheader('Cargar Datos')
    archivo = st.file_uploader('Cargar Archivo CSV o XLS', type=['csv', 'xls'])
    if archivo:
        df = cargar_datos(archivo)
        if df is not None:
            st.session_state.df = df
            st.warning('El archivo se ha cargado correctamente')
            st.write('El dataset tiene', df.shape[0], 'filas y', df.shape[1], 'columnas')
            st.write(df)
    else:
        st.warning('Por favor carga un archivo')

def mostrar_analisis_datos():
    st.subheader('Analisis de Datos')
    if 'df' not in st.session_state:
        st.warning('Por favor carga un archivo')
    else:
        df = st.session_state.df
        st.subheader('Primeras 5 filas')
        st.write(df.head())
        st.subheader('Información descriptiva del dataset')
        st.write(df.describe())