import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import altair as alt
from datetime import datetime

# Configuración de la página
st.set_page_config(page_title="Análisis de Consumo Eléctrico - K-means")

def load_data(file):
    # Cargar el archivo CSV
    df = pd.read_csv(file)
    
    # Convertir la columna timestamp a datetime si es necesario
    try:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
    except:
        st.error("Asegúrate que la columna de tiempo se llame 'timestamp'")
        return None
    
    return df

def prepare_data(df):
    # Extraer características temporales
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    
    # Preparar datos para clustering
    X = df[['kwh', 'hour', 'dayofweek']].copy()
    
    # Escalar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, X

def perform_clustering(X_scaled, n_clusters):
    # Aplicar K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    return clusters

# Título de la aplicación
st.title("📊 Análisis de Consumo Eléctrico con K-means")

# Sidebar para configuración
st.sidebar.header("Configuración")
uploaded_file = st.sidebar.file_uploader("Cargar archivo CSV", type=['csv'])
n_clusters = st.sidebar.slider("Número de clusters", min_value=2, max_value=10, value=3)

if uploaded_file is not None:
    # Cargar y procesar datos
    df = load_data(uploaded_file)
    
    if df is not None:
        # Preparar datos para clustering
        X_scaled, X = prepare_data(df)
        
        # Realizar clustering
        clusters = perform_clustering(X_scaled, n_clusters)
        
        # Añadir clusters al dataframe original
        df['Cluster'] = clusters.astype(str)  # Convertir a string para mejor visualización
        
        # Mostrar información de clusters
        st.header("📑 Resumen de Clusters")
        cluster_summary = df.groupby('Cluster').agg({
            'kwh': ['mean', 'min', 'max', 'count']
        }).round(2)
        
        cluster_summary.columns = ['Consumo Promedio (kWh)', 'Consumo Mínimo (kWh)', 
                                 'Consumo Máximo (kWh)', 'Número de Registros']
        st.dataframe(cluster_summary)
        
        # Visualización de clusters
        st.header("📈 Visualización de Clusters")
        
        # Crear dos columnas para los gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dispersión por Hora del Día")
            # Gráfico de dispersión usando Altair
            scatter_chart = alt.Chart(df).mark_circle().encode(
                x=alt.X('hour:Q', title='Hora del Día'),
                y=alt.Y('kwh:Q', title='Consumo (kWh)'),
                color=alt.Color('Cluster:N', title='Cluster'),
                tooltip=['hour', 'kwh', 'Cluster']
            ).properties(
                width=300,
                height=300
            ).interactive()
            
            st.altair_chart(scatter_chart, use_container_width=True)
        
        with col2:
            st.subheader("Consumo por Día de la Semana")
            # Gráfico de dispersión por día de la semana
            weekday_chart = alt.Chart(df).mark_circle().encode(
                x=alt.X('dayofweek:Q', title='Día de la Semana (0=Lunes)'),
                y=alt.Y('kwh:Q', title='Consumo (kWh)'),
                color=alt.Color('Cluster:N', title='Cluster'),
                tooltip=['dayofweek', 'kwh', 'Cluster']
            ).properties(
                width=300,
                height=300
            ).interactive()
            
            st.altair_chart(weekday_chart, use_container_width=True)
        
        # Gráfico de línea temporal
        st.subheader("Consumo a lo Largo del Tiempo por Cluster")
        line_chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X('timestamp:T', title='Fecha y Hora'),
            y=alt.Y('kwh:Q', title='Consumo (kWh)'),
            color=alt.Color('Cluster:N', title='Cluster'),
            tooltip=['timestamp', 'kwh', 'Cluster']
        ).properties(
            width=800,
            height=400
        ).interactive()
        
        st.altair_chart(line_chart, use_container_width=True)
        
        # Mostrar estadísticas básicas
        st.header("📊 Estadísticas por Cluster")
        
        # Crear gráfico de caja usando Altair
        box_plot = alt.Chart(df).mark_boxplot().encode(
            x=alt.X('Cluster:N', title='Cluster'),
            y=alt.Y('kwh:Q', title='Consumo (kWh)'),
            color=alt.Color('Cluster:N', title='Cluster')
        ).properties(
            width=600,
            height=400
        )
        
        st.altair_chart(box_plot, use_container_width=True)
        
        # Descargar resultados
        st.header("💾 Descargar Resultados")
        csv = df.to_csv(index=False)
        st.download_button(
            label="Descargar resultados como CSV",
            data=csv,
            file_name="resultados_clustering.csv",
            mime="text/csv"
        )
        
else:
    st.info("👆 Por favor, carga un archivo CSV para comenzar el análisis.")
    st.markdown("""
    El archivo CSV debe contener las siguientes columnas:
    - `timestamp`: Fecha y hora de la medición
    - `kwh`: Consumo eléctrico en kilovatios-hora
    
    Los datos serán procesados y clasificados automáticamente usando el algoritmo K-means.
    """)

# Agregar información sobre el uso
st.sidebar.markdown("""
---
### Información de Uso
- Los datos se procesan automáticamente al cargar el archivo
- Puedes ajustar el número de clusters usando el deslizador
- Los gráficos son interactivos: prueba a hacer zoom o pasar el mouse sobre los puntos
""")
