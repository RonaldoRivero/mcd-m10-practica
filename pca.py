import streamlit as st
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
def mostrar_pca():
    st.subheader('PCA')
    if 'df_normalizado' not in st.session_state:
        st.warning('Por favor carga un archivo y normalizalo')
    else:
        df_normalizado = st.session_state.df_normalizado

        pca = PCA()
        datos_pca = pca.fit_transform(df_normalizado)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write('Varianza Explicada:', pca.explained_variance_)
        with col2:
            st.write('Varianza Explicada:', pca.explained_variance_ratio_)
        with col3:
            varianza_acumulada = np.cumsum(pca.explained_variance_ratio_)
            st.write('Varianza Acumulada:', varianza_acumulada)

        fig = plt.figure(figsize=(8, 8))
        plt.plot(varianza_acumulada)
        plt.hlines(y=0.9, xmin=0, xmax=4, colors='r', linestyles='--')
        plt.xlabel('Numero de Componentes Principales')
        plt.ylabel('Varianza Acumulada')
        st.pyplot(fig)
