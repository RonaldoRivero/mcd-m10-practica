import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title='Trabajo Práctico'
                   , page_icon=':shark:'
                   , layout='wide')
st.title('Trabajo Práctico')

st.sidebar.title('Menu de Opciones')
opciones = ["Cargar Datos",
            'Analisis de Datos',
            'Limpiar y Preparar Datos',
            'PCA',
            'Cluster']
opcion = st.sidebar.radio('Selecciona una Opcion', opciones)

@st.cache_data
def cargar_datos(archivo):
    if archivo.name.endswith('csv'):
        df = pd.read_csv(archivo)
    elif archivo.name.endswith('xls'):
        df = pd.read_excel(archivo)
    else:
        raise Exception('Este formato no es compatible')
    return df


if opcion == 'Cargar Datos':
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

elif opcion == 'Analisis de Datos':
    st.subheader('Analisis de Datos')
    if 'df' not in st.session_state:
        st.warning('Por favor carga un archivo')
    else:
        df = st.session_state.df
        st.subheader('primeras 5 filas')
        st.write(df.head())
        st.subheader('Informacion del dataframe')
        st.write(df.describe())

elif opcion == 'Limpiar y Preparar Datos':
    st.subheader('Limpiar y Preparar Datos')
    if 'df' not in st.session_state:
        st.warning('Por favor carga un archivo')
    else:
        df = st.session_state.df
        st.write(df.head())
        st.subheader('Limpiar Datos')
        columnas = df.columns.tolist()
        columnas_a_eliminar = st.multiselect("Selecciona las columnas que deseas eliminar", columnas)
        if columnas_a_eliminar:
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
        limpiar_datos = st.radio('Deseas normaliazar valores', ['Si', 'No'])
        if limpiar_datos == 'Si':
            if st.button('Confirmar'):
                with st.spinner('Normalizando valores'):
                    df.dropna(inplace=True)
                    st.write('Valores normalizados')
                    df_normalizado = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)

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

elif opcion == 'PCA':
    st.subheader('PCA')
    if 'df_normalizado' not in st.session_state:
        st.warning('Por favor carga un archivo y normalizalo')
    else:
        df_normalizado = st.session_state.df_normalizado

        pca = PCA()
        datos_pca = pca.fit_transform(df_normalizado)

        st.write('Varianza Explicada:', pca.explained_variance_)
        st.write('Varianza Explicada:', pca.explained_variance_ratio_)

        varianza_acumulada = np.cumsum(pca.explained_variance_ratio_)
        st.write('Varianza Acumulada:', varianza_acumulada)

        fig = plt.figure(figsize=(8, 8))
        plt.plot(varianza_acumulada)
        plt.hlines(y=0.9, xmin=0, xmax=4, colors='r', linestyles='--')
        plt.xlabel('Numero de Componentes Principales')
        plt.ylabel('Varianza Acumulada')
        st.pyplot(fig)

elif opcion == 'Cluster':
    st.subheader("Clustering - KMeans")
    if 'df_normalizado' not in st.session_state:
        st.warning('Por favor carga un archivo y normalizalo')
    else:
        df_normalizado = st.session_state.df_normalizado

        n_clusters = st.slider('Número de Clusters', 2, 10, 3)
        random_state = st.slider('Random State', 0, 100, 42)
        max_iter = st.slider('Numero de Iteraciones', 100, 1000, 300)

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, max_iter=max_iter).fit(
            df_normalizado.values)

        silhouette = silhouette_score(df_normalizado.values, kmeans.labels_)

        col1, col2 = st.columns(2)
        columnas = df_normalizado.columns.tolist()
        with col1:
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(df_normalizado.iloc[:, 0], df_normalizado.iloc[:, 1], c=kmeans.labels_, s=150,
                        marker='8')
            plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=350, marker='+')
            plt.xlabel(columnas[0])
            plt.ylabel(columnas[1])
            plt.title('KMeans Clustering')
            st.pyplot(fig)
        with col2:
            st.markdown('## Medidas de Evaluación')
            st.write('Numero de Clusters:', kmeans.n_clusters)
            st.write('Inercia:', kmeans.inertia_)
            st.write('Silhouette Score:', silhouette)

        st.subheader('Método del Codo')
        inertias = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=random_state).fit(df_normalizado.values)
            inertias.append(kmeans.inertia_)

        fig = plt.figure(figsize=(6, 6))
        plt.scatter(range(1, 11), inertias, c='red', s=150, marker='o', edgecolor='black')
        plt.plot(range(1, 11), inertias)
        plt.xlabel('Número de Clusters')
        plt.ylabel('Inertia')
        plt.title('Método del Codo')
        st.pyplot(fig)
