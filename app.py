import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from pca import mostrar_pca
from kmean import mostrar_cluster_kmeans
from preparacion_datos import mostrar_cargar_datos
from preparacion_datos import mostrar_analisis_datos

st.set_page_config(page_title='Trabajo Práctico'
                   , page_icon=':shark:'
                   , layout='wide')
st.title('Trabajo Práctico')

st.sidebar.title('Menu de Opciones')
opciones = ["Cargar Datos",
            'Analisis de Datos',
            'Limpiar y Preparar Datos',
            'PCA',
            'KMeans']
opcion = st.sidebar.radio('Selecciona una Opcion', opciones)


def mostrar_limpiar_preparar_datos():
    st.subheader('Limpiar y Preparar Datos')
    if 'df' not in st.session_state:
        st.warning('Por favor carga un archivo')
    else:
        df = st.session_state.df
        st.write(df.head())
        st.subheader('Limpiar Datos')
        columnas = df.columns.tolist()
        columnas_a_eliminar = st.multiselect("Selecciona las columnas que deseas eliminar", columnas)
        if len(columnas_a_eliminar) > 0:
            if st.button('Eliminar Columna'):
                with st.spinner('Eliminando columnas'):
                    df = df.drop(columns=columnas_a_eliminar)
                    st.write("datos después de eliminar las columnas seleccionadas:")
                    st.write(df.head())

                st.session_state.df = df
        else:
            st.write("No se han seleccionado columnas para eliminar.")

        st.subheader('Valores Nulos')
        if df.isnull().sum().sum() > 0:
            st.warning('El dataset tiene valores nulos')
            st.write(df.isnull().sum())
            limpiar_datos = st.radio('Deseas eliminar los valores nulos', ['Si', 'No'])
            if limpiar_datos == 'Si':
                if st.button('Confirmar'):
                    with st.spinner('Eliminando valores nulos'):
                        df.dropna(inplace=True)
                        st.write('Valores nulos eliminados')
                        st.write(df.isnull().sum())

                    st.session_state.df = df
                    st.success('Valores nulos eliminados')
                    st.write(df.head())
            else:
                st.info('No se eliminaron los valores nulos')
        else:
            st.success('No hay valores nulos en el dataset')

        st.subheader('Preparar Datos')
        columnas = df.columns.tolist()
        columnas_select = st.multiselect("Selecciona las columnas", columnas)
        if len(columnas_select) == 2:
            fig = plt.figure(figsize=(8, 8))
            plt.title('Antes de Normalizar', fontsize=15)
            plt.scatter(df[columnas_select[0]], df[columnas_select[1]],
                        marker='8', s=500, c='purple', alpha=0.5)
            plt.xlabel(columnas_select[0], fontsize=12)
            plt.ylabel(columnas_select[1], fontsize=12)
            st.pyplot(fig)
        else:
            st.write("No ha seleccionado columnas.")

        st.write('Normalizar Columnas')
        limpiar_datos = st.radio('Deseas normaliazar valores', ['StandarScaler', 'MaxMin', 'No'])
        if limpiar_datos == 'StandarScaler':
            if st.button('Normalizar StandarScaler'):
                with st.spinner('Normalizando valores'):
                    st.write('Valores normalizados')
                    df_normalizado = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)

                st.session_state.df_normalizado = df_normalizado
                st.success('Valores normalizados')
                st.write(df_normalizado.head())
        elif limpiar_datos == 'MaxMin':
            if st.button('Normalizar MaxMin'):
                with st.spinner('Normalizando valores'):
                    st.write('Valores normalizados')
                    scaler = MinMaxScaler()
                    df_normalizado = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

                st.session_state.df_normalizado = df_normalizado
                st.success('Valores normalizados')
                st.write(df_normalizado.head())
        else:
            st.info('No se eliminaron los valores nulos')

        if 'df_normalizado' in st.session_state:
            df_normalizado = st.session_state.df_normalizado
            columnas = df_normalizado.columns.tolist()
            columnas_normal_select = st.multiselect("Selecciona las columnas normalizada", columnas)
            if len(columnas_normal_select) == 2:
                fig = plt.figure(figsize=(8, 8))
                plt.title('Despues de Normalizar', fontsize=15)
                plt.scatter(df_normalizado[columnas_normal_select[0]], df_normalizado[columnas_normal_select[1]],
                            marker='8', s=500, c='red', alpha=0.5)
                plt.xlabel(columnas_normal_select[0], fontsize=12)
                plt.ylabel(columnas_normal_select[1], fontsize=12)
                st.pyplot(fig)
            else:
                st.write("No ha seleccionado columnas.")

if opcion == 'Cargar Datos':
    mostrar_cargar_datos()

elif opcion == 'Analisis de Datos':
    mostrar_analisis_datos()

elif opcion == 'Limpiar y Preparar Datos':
    mostrar_limpiar_preparar_datos()

elif opcion == 'PCA':
    mostrar_pca()

elif opcion == 'KMeans':
    mostrar_cluster_kmeans()
