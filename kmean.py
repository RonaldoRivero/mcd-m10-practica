import streamlit as st
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def mostrar_cluster_kmeans():
    st.subheader("KMeans - Clustering ")
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