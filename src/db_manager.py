import sqlite3
import pandas as pd
from datetime import datetime
import os

# Definir la ruta de la DB relativa a este script
# Se guardará en la carpeta 'data' si existe, si no en la raíz.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'consultations.db')

def get_connection():
    """Establece conexión con la base de datos SQLite."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_db():
    """Inicializa la tabla si no existe."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS consultations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                sex TEXT,
                age INTEGER,
                hgb REAL,
                wbc REAL,
                plt REAL,
                mcv REAL,
                prediction TEXT,
                confidence REAL,
                status TEXT
            )
        ''')
        conn.commit()
    except Exception as e:
        print(f"Error inicializando DB: {e}")
    finally:
        conn.close()

def save_consultation(inputs: dict, prediction: str, confidence: float, status: str):
    """Guarda una nueva consulta."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Extraer datos del diccionario de inputs
        # Aseguramos que existan las claves o ponemos valores por defecto
        sex = 'M' if inputs.get('Sex') == 1 else 'F'
        age = inputs.get('Age', 0)
        hgb = inputs.get('HGB', 0.0)
        wbc = inputs.get('WBC', 0.0)
        plt_val = inputs.get('PLT', 0.0)
        mcv = inputs.get('MCV', 0.0)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute('''
            INSERT INTO consultations (timestamp, sex, age, hgb, wbc, plt, mcv, prediction, confidence, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, sex, age, hgb, wbc, plt_val, mcv, prediction, confidence, status))
        
        conn.commit()
    except Exception as e:
        print(f"Error guardando consulta: {e}")
    finally:
        conn.close()

def get_history(limit: int = 50) -> pd.DataFrame:
    """Recupera el historial de consultas."""
    conn = get_connection()
    try:
        query = f"SELECT * FROM consultations ORDER BY id DESC LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        
        # Renombrar columnas para la UI
        df = df.rename(columns={
            'id': 'ID',
            'timestamp': 'Fecha',
            'sex': 'Sexo',
            'age': 'Edad',
            'hgb': 'HGB (g/dL)',
            'wbc': 'WBC (k/µL)',
            'prediction': 'Diagnóstico',
            'confidence': 'Confianza (%)',
            'status': 'Estado'
        })
        return df
    except Exception as e:
        # Si falla (ej. tabla no existe aún), devolvemos DF vacío
        return pd.DataFrame()
    finally:
        conn.close()