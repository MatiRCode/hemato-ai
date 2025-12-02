import streamlit as st
import pandas as pd
from datetime import datetime

# ==========================================
# GESTOR DE BASE DE DATOS EN MEMORIA (RAM)
# ==========================================
# Solución robusta para Streamlit Cloud donde la escritura en disco
# suele estar restringida o ser efímera.

def init_db():
    """Inicializa la estructura de almacenamiento en la sesión del usuario."""
    if 'consultation_history' not in st.session_state:
        st.session_state['consultation_history'] = []

def save_consultation(inputs: dict, prediction: str, confidence: float, status: str):
    """Guarda el registro en la lista de memoria."""
    # Aseguramos que la DB esté iniciada
    init_db()
    
    # Crear el registro estructurado
    record = {
        'ID': len(st.session_state['consultation_history']) + 1,
        'Fecha': datetime.now().strftime("%H:%M:%S"), # Solo hora para sesión rápida
        'Sexo': 'M' if inputs.get('Sex') == 1 else 'F',
        'Edad': int(inputs.get('Age', 0)),
        'HGB': float(inputs.get('HGB', 0.0)),
        'WBC': float(inputs.get('WBC', 0.0)),
        'Diagnóstico IA': prediction,
        'Confianza': f"{confidence:.1f}%",
        'Alerta': status
    }
    
    # Insertar al principio de la lista (LIFO) para que salga el más nuevo arriba
    st.session_state['consultation_history'].insert(0, record)

def get_history(limit: int = 50) -> pd.DataFrame:
    """Convierte la lista de memoria en un DataFrame para visualizar."""
    init_db()
    
    data = st.session_state['consultation_history']
    
    if not data:
        return pd.DataFrame()
        
    # Convertir a DataFrame y recortar al límite
    df = pd.DataFrame(data)
    return df.head(limit)