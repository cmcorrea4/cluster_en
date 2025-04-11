
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Configuración simple de la página
st.set_page_config(page_title="Análisis de Consumo Eléctrico", layout="wide")

# Título de la aplicación
st.title("📊 Análisis de Consumo Eléctrico")
st.write("Versión simplificada para prueba de despliegue")

# Barra lateral para configuración
with st.sidebar:
    st.header("Configuración")
    uploaded_file = st.file_uploader("Cargar archivo CSV", type=['csv'])
    
    st.markdown("""
    ---
    ### Información de Uso
    - Esta es una versión simplificada para pruebas
    - Carga un archivo CSV para ver datos básicos
    """)

# Contenido principal
if uploaded_file is not None:
    try:
        # Cargar archivo CSV
        df = pd.read_csv(uploaded_file)
        
        # Mostrar vista previa
        st.subheader("Vista previa de los datos")
        st.write(df.head())
        
        # Estadísticas básicas
        st.subheader("Estadísticas básicas")
        st.write(df.describe())
        
    except Exception as e:
        st.error(f"Error al procesar el archivo: {str(e)}")
else:
    st.info("👆 Por favor, carga un archivo CSV para comenzar el análisis.")
    
    # Opción de datos de muestra
    if st.button("Usar datos de muestra para prueba"):
        # Crear datos de muestra simples
        fechas = pd.date_range(start='2024-01-01', periods=24, freq='H')
        valores_kwh = np.random.normal(0.2, 0.1, size=24)
        
        datos_muestra = pd.DataFrame({
            'Datetime': fechas,
            'Kwh': valores_kwh
        })
        
        # Mostrar datos de muestra
        st.subheader("Datos de muestra")
        st.write(datos_muestra)
