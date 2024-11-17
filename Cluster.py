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
    df = pd.read_csv(file, index_col=0)  # Usar la primera columna como índice
    
    # Mostrar información sobre las columnas disponibles
    st.write("Columnas disponibles en el archivo:", df.columns.tolist())
    
    # Convertir la columna Datetime a datetime si es necesario
    try:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
    except KeyError:
        st.error("No se encontró la columna 'Datetime' en el archivo")
        return None
    except Exception as e:
        st.error(f"Error al procesar la columna Datetime: {str(e)}")
        return None
    
    # Verificar si existe la columna Kwh
    if 'Kwh' not in df.columns:
        st.error("No se encontró la columna 'Kwh' en el archivo")
        return None
    
    # Eliminar filas con valores faltantes
    df = df.dropna()
    
    return df

def prepare_data(df):
    try:
        # Extraer características temporales
        df['hour'] = df['Datetime'].dt.hour
        df['dayofweek'] = df['Datetime'].dt.dayofweek
        
        # Preparar datos para clustering
        X = df[['Kwh', 'hour', 'dayofweek']].copy()
        
        # Escalar los datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, X
    except Exception as e:
        st.error(f"Error al preparar los datos: {str(e)}")
        st.write("Estructura actual del DataFrame:")
        st.write(df.head())
        st.write("Tipos de datos de las columnas:")
        st.write(df.dtypes)
        return None, None

def perform_clustering(X_scaled, n_clusters):
    try:
        # Aplicar K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        return clusters
    except Exception as e:
        st.error(f"Error al realizar el clustering: {str(e)}")
        return None

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
        # Mostrar las primeras filas de los datos
        st.subheader("Vista previa de los datos")
        st.write(df.head())
        
        # Preparar datos para clustering
        X_scaled, X = prepare_data(df)
        
        if X_scaled is not None and X is not None:
            # Realizar clustering
            clusters = perform_clustering(X_scaled, n_clusters)
            
            if clusters is not None:
                # Añadir clusters al dataframe original
                df['Cluster'] = clusters.astype(str)
                
                # Mostrar información de clusters
                st.header("📑 Resumen de Clusters")
                cluster_summary = df.groupby('Cluster').agg({
                    'Kwh': ['mean', 'min', 'max', 'count']
                }).round(2)
                
                cluster_summary.columns = ['Consumo Promedio (kWh)', 'Consumo Mínimo (kWh)', 
                                         'Consumo Máximo (kWh)', 'Número de Registros']
                st.dataframe(cluster_summary)
                
                # Visualización de clusters
                st.header("📈 Visualización de Clusters")
                
                try:
                    # Crear dos columnas para los gráficos
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Dispersión por Hora del Día")
                        scatter_chart = alt.Chart(df).mark_circle().encode(
                            x=alt.X('hour:Q', title='Hora del Día'),
                            y=alt.Y('Kwh:Q', title='Consumo (kWh)'),
                            color=alt.Color('Cluster:N', title='Cluster'),
                            tooltip=['hour', 'Kwh', 'Cluster']
                        ).properties(
                            width=300,
                            height=300
                        ).interactive()
                        
                        st.altair_chart(scatter_chart, use_container_width=True)
                    
                    with col2:
                        st.subheader("Consumo por Día de la Semana")
                        weekday_chart = alt.Chart(df).mark_circle().encode(
                            x=alt.X('dayofweek:Q', 
                                   title='Día de la Semana',
                                   scale=alt.Scale(domain=[0, 6]),
                                   axis=alt.Axis(
                                       values=[0, 1, 2, 3, 4, 5, 6],
                                       labelExpr="['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'][datum.value]"
                                   )),
                            y=alt.Y('Kwh:Q', title='Consumo (kWh)'),
                            color=alt.Color('Cluster:N', title='Cluster'),
                            tooltip=['dayofweek', 'Kwh', 'Cluster']
                        ).properties(
                            width=300,
                            height=300
                        ).interactive()
                        
                        st.altair_chart(weekday_chart, use_container_width=True)
                    
                    # Gráfico de línea temporal
                    st.subheader("Consumo a lo Largo del Tiempo por Cluster")
                    line_chart = alt.Chart(df).mark_line(point=True).encode(
                        x=alt.X('Datetime:T', title='Fecha y Hora'),
                        y=alt.Y('Kwh:Q', title='Consumo (kWh)'),
                        color=alt.Color('Cluster:N', title='Cluster'),
                        tooltip=['Datetime', 'Kwh', 'Cluster']
                    ).properties(
                        width=800,
                        height=400
                    ).interactive()
                    
                    st.altair_chart(line_chart, use_container_width=True)
                    
                    # Gráfico de caja
                    st.header("📊 Estadísticas por Cluster")
                    box_plot = alt.Chart(df).mark_boxplot().encode(
                        x=alt.X('Cluster:N', title='Cluster'),
                        y=alt.Y('Kwh:Q', title='Consumo (kWh)'),
                        color=alt.Color('Cluster:N', title='Cluster')
                    ).properties(
                        width=600,
                        height=400
                    )
                    
                    st.altair_chart(box_plot, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error al crear las visualizaciones: {str(e)}")
                
                # Descargar resultados
                st.header("💾 Descargar Resultados")
                csv = df.to_csv(index=True)
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
    - `Datetime`: Fecha y hora de la medición
    - `Kwh`: Consumo eléctrico en kilovatios-hora
    
    Formato esperado:
    ```
    ,Datetime,Kwh
    0,2024-11-01 00:00:00,0.14625
    1,2024-11-01 01:00:00,0.12281
    ...
    ```
    """)

# Agregar información sobre el uso
st.sidebar.markdown("""
---
### Información de Uso
- Los datos se procesan automáticamente al cargar el archivo
- Puedes ajustar el número de clusters usando el deslizador
- Los gráficos son interactivos: prueba a hacer zoom o pasar el mouse sobre los puntos
""")
