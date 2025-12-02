import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import joblib
import os
import sys

# ==========================================
# 0. INGENIERÍA DE RUTAS
# ==========================================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
SRC_DIR = os.path.join(ROOT_DIR, 'src')
ASSETS_DIR = os.path.join(ROOT_DIR, 'assets')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

try:
    from db_manager import init_db, save_consultation, get_history
except ImportError:
    def init_db(): pass
    def save_consultation(*args): pass
    def get_history(*args): return pd.DataFrame()

# ==========================================
# 1. CONFIGURACIÓN DEL SISTEMA
# ==========================================
st.set_page_config(
    page_title="HematoAI CDSS v1.1",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded" # Fuerza sidebar abierta
)

init_db()

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#00000000', 
    'axes.facecolor': '#00000000',
    'text.color': '#cbd5e1',         
    'axes.labelcolor': '#64748b',    
    'xtick.color': '#475569',
    'ytick.color': '#475569',
    'grid.color': '#334155',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Inter', 'Arial']
})

css_file = os.path.join(ASSETS_DIR, 'style.css')
if os.path.exists(css_file):
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# ==========================================
# 2. MOTOR DE INFERENCIA
# ==========================================
@st.cache_resource
def load_engine():
    model_path = os.path.join(MODELS_DIR, "xgboost_clinical_v2.json")
    encoder_path = os.path.join(MODELS_DIR, "label_encoder_v2.joblib")
    anomaly_path = os.path.join(MODELS_DIR, "anomaly_detector_v2.joblib")

    model = xgb.XGBClassifier()
    try:
        model.load_model(model_path)
    except Exception as e:
        st.error(f"❌ Error Crítico: Modelo no encontrado en {model_path}. Detalles: {e}")
        st.stop()
    
    try:
        le = joblib.load(encoder_path)
        class_names = le.classes_
    except:
        st.error("❌ Error Crítico: Label Encoder no encontrado.")
        st.stop()
        
    try:
        anomaly_model = joblib.load(anomaly_path)
    except:
        anomaly_model = None

    try:
        explainer = shap.TreeExplainer(model)
    except:
        explainer = None
        
    return model, class_names, explainer, anomaly_model

model, class_names, explainer, anomaly_model = load_engine()

# ==========================================
# 3. INTERFAZ: SIDEBAR
# ==========================================
st.sidebar.markdown("### 🧬 HEMATO-AI")
st.sidebar.caption("v1.1 Production Release")
st.sidebar.markdown("---")

def get_user_input():
    st.sidebar.subheader("Paciente")
    c1, c2 = st.sidebar.columns(2)
    sex = c1.selectbox("Sexo", ["M", "F"], index=0)
    age = c2.number_input("Edad (años)", 0, 120, 45)
    sex_val = 1 if sex == "M" else 2

    st.sidebar.subheader("Hemograma (Serie Roja/Blanca)")
    wbc = st.sidebar.number_input("Leucocitos (WBC k/µL)", 0.0, 500.0, 7.5, 0.1)
    hgb = st.sidebar.number_input("Hemoglobina (HGB g/dL)", 0.0, 25.0, 14.0, 0.1)
    plt_cnt = st.sidebar.number_input("Plaquetas (PLT k/µL)", 0.0, 5000.0, 250.0, 1.0)
    mcv = st.sidebar.number_input("VCM (fL)", 0.0, 200.0, 88.0, 0.1)

    with st.sidebar.expander("🔬 Diferencial y Frotis", expanded=False):
        neu = st.number_input("Neutrófilos Abs.", 0.0, 100.0, 4.0, 0.1)
        lym = st.number_input("Linfocitos Abs.", 0.0, 100.0, 2.0, 0.1)
        hct = st.number_input("Hematocrito (%)", 0.0, 100.0, 42.0, 0.1)
        rbc = st.number_input("Eritrocitos (RBC M/µL)", 0.0, 15.0, 4.8, 0.1)
        mch = st.number_input("HCM (pg)", 0.0, 100.0, 30.0, 0.1)

    full_data = {
        'Sex': sex_val, 'Age': age, 'WBC': wbc, 'NEU': neu, 'LYM': lym,
        'RBC': rbc, 'HGB': hgb, 'HCT': hct, 'MCV': mcv, 'MCH': mch, 'PLT': plt_cnt,
        'NEU_percent': (neu/wbc*100) if wbc > 0 else 0,
        'LYM_percent': (lym/wbc*100) if wbc > 0 else 0
    }
    
    try:
        model_features = model.get_booster().feature_names
        if model_features:
            filtered_data = {k: v for k, v in full_data.items() if k in model_features}
            features = pd.DataFrame([filtered_data])[model_features]
        else:
            features = pd.DataFrame(full_data, index=[0])
    except:
        features = pd.DataFrame(full_data, index=[0])
        
    return features

input_df = get_user_input()

st.sidebar.markdown("---")
st.sidebar.info("System Status: 🟢 ONLINE")

# ==========================================
# 4. DASHBOARD CLÍNICO
# ==========================================

tab_diag, tab_hist = st.tabs(["EVALUACIÓN DIAGNÓSTICA", "HISTORIAL DE CASOS"])

with tab_diag:
    hgb_val = input_df['HGB'].values[0] if 'HGB' in input_df.columns else 0
    if hgb_val > 0 and hgb_val < 2.0:
        st.error("⛔ ERROR: Hemoglobina < 2.0 g/dL. Valor crítico incompatible o error de entrada.")
        st.stop()

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🔍 Análisis de Riesgo")
        
        if st.button("PROCESAR BIOPSIA DIGITAL", type="primary", use_container_width=True):
            with st.spinner("Ejecutando pipeline de inferencia v1.1..."):
                
                prediction = model.predict(input_df)
                pred_idx = int(prediction[0])
                pred_class = class_names[pred_idx]
                
                probas = model.predict_proba(input_df)[0]
                confidence = probas[pred_idx] * 100
                
                is_anomaly = False
                if anomaly_model:
                    try:
                        if anomaly_model.predict(input_df)[0] == -1:
                            is_anomaly = True
                    except:
                        pass 

                status_color = "#64748b"
                status_icon = "⚪"
                
                if is_anomaly:
                    if confidence > 50:
                        status_color = "#f59e0b"
                        status_icon = "⚠️"
                        ui_message = f"VALORES ATÍPICOS DETECTADOS - PERO SE PROYECTA: {pred_class}"
                    else:
                        status_color = "#ef4444"
                        status_icon = "⛔"
                        pred_class = "PATRÓN NO RECONOCIDO"
                        confidence = 0.0
                        ui_message = "PERFIL FUERA DE RANGO ESTADÍSTICO - REQUIERE REVISIÓN MANUAL"
                else:
                    if "Normal" in pred_class:
                        status_color = "#10b981"
                        status_icon = "✅"
                        ui_message = "PARÁMETROS DENTRO DE LÍMITES FISIOLÓGICOS"
                    elif "Leucemia" in pred_class:
                        status_color = "#f43f5e"
                        status_icon = "🚨"
                        ui_message = "MARCADORES COMPATIBLES CON SÍNDROME LINFOPROLIFERATIVO"
                    elif "Anemia" in pred_class:
                        status_color = "#f97316"
                        status_icon = "📉"
                        ui_message = "DÉFICIT ERITROCITARIO DETECTADO"
                    else:
                        status_color = "#8b5cf6"
                        status_icon = "🦠"
                        ui_message = "REACTIVIDAD INMUNE / INFECCIOSA"

                save_consultation(input_df.iloc[0].to_dict(), pred_class, confidence, "OK" if not is_anomaly else "WARN")

                st.markdown(f"""
                <div style="border-left: 4px solid {status_color}; padding-left: 20px; margin-top: 20px;">
                    <h5 style="color: {status_color}; margin:0; letter-spacing: 1px;">RESULTADO DEL MODELO</h5>
                    <h1 style="font-size: 2.8rem; margin: 5px 0; color: #f8fafc;">{pred_class}</h1>
                    <div style="display: flex; align-items: center; gap: 10px; margin-top: 10px;">
                        <span style="background: {status_color}33; color: {status_color}; padding: 4px 12px; border-radius: 99px; font-weight: bold; font-size: 0.9rem; border: 1px solid {status_color}66;">
                            {status_icon} Confianza: {confidence:.1f}%
                        </span>
                    </div>
                    <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 15px; font-family: monospace;">
                        > {ui_message}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("###### 📊 Probabilidades por Clase")
                probs_df = pd.DataFrame({'Clase': class_names, 'Prob': probas}).sort_values('Prob', ascending=False).head(4)
                for _, row in probs_df.iterrows():
                    col_txt, col_bar = st.columns([1.5, 3])
                    with col_txt: st.caption(f"{row['Clase']}")
                    with col_bar: st.progress(row['Prob'])

        else:
            st.info("Configure los parámetros en el panel lateral y ejecute el análisis.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown("#### 🧬 Evidencia Clínica")
        
        if 'pred_class' in locals() and confidence > 0:
            
            try:
                st.markdown('<div class="glass-card" style="padding: 15px;">', unsafe_allow_html=True)
                st.markdown("**Drivers de la Decisión (SHAP)**")
                st.caption("¿Qué variables inclinaron el diagnóstico?")
                
                if explainer:
                    shap_vals = explainer.shap_values(input_df)
                    
                    if isinstance(shap_vals, list):
                        raw_shap = shap_vals[pred_idx]
                    else:
                        raw_shap = shap_vals

                    if not isinstance(raw_shap, np.ndarray):
                        raw_shap = np.array(raw_shap)
                    sv = raw_shap.reshape(-1)
                    
                    feature_names = input_df.columns.tolist()
                    
                    if len(sv) > len(feature_names) and len(feature_names) > 0:
                        try:
                            n_classes = int(len(sv) / len(feature_names))
                            sv_matrix = sv.reshape(n_classes, len(feature_names))
                            sv = sv_matrix[pred_idx]
                        except:
                            pass 

                    if len(feature_names) == len(sv):
                        shap_data = pd.DataFrame({
                            'Feature': feature_names,
                            'SHAP': sv.tolist()
                        })
                        shap_data['Abs'] = shap_data['SHAP'].abs()
                        shap_data = shap_data.sort_values('Abs', ascending=False).head(5)
                        
                        fig, ax = plt.subplots(figsize=(4, 3))
                        bars = ax.barh(shap_data['Feature'], shap_data['SHAP'], color=status_color)
                        ax.axvline(0, color='#475569', linewidth=0.8)
                        ax.set_facecolor('#00000000')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                        ax.bar_label(bars, fmt='%.2f', padding=3, color='white', fontsize=8)
                        st.pyplot(fig)
                    else:
                        st.warning(f"⚠️ Discrepancia dimensional no resuelta (Cols={len(feature_names)}, Val={len(sv)}).")
                else:
                    st.warning("Explainer no inicializado.")
                    
                st.markdown('</div>', unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error visualizando SHAP: {str(e)}")

            try:
                st.markdown('<div class="glass-card" style="padding: 15px;">', unsafe_allow_html=True)
                st.markdown("**Morfología Vectorial**")
                
                def get_val(col, div): return min(input_df.get(col, [0]).values[0] / div, 1.0)
                
                radar_vals = [
                    get_val('WBC', 20.0),
                    get_val('HGB', 18.0),
                    get_val('PLT', 450.0),
                    get_val('MCV', 110.0),
                    get_val('NEU_percent', 100.0)
                ]
                radar_labels = ['WBC', 'HGB', 'PLT', 'MCV', 'NEU%']
                
                fig_r = plt.figure(figsize=(3, 3))
                ax_r = fig_r.add_subplot(111, polar=True)
                angles = np.linspace(0, 2*np.pi, len(radar_labels), endpoint=False).tolist()
                radar_vals += radar_vals[:1]
                angles += angles[:1]
                ax_r.fill(angles, radar_vals, color=status_color, alpha=0.3)
                ax_r.plot(angles, radar_vals, color=status_color, linewidth=2)
                ax_r.set_xticks(angles[:-1])
                ax_r.set_xticklabels(radar_labels, size=7, color='#94a3b8')
                ax_r.set_yticklabels([])
                ax_r.spines['polar'].set_visible(False)
                ax_r.grid(color='#334155', alpha=0.5)
                st.pyplot(fig_r)
                st.markdown('</div>', unsafe_allow_html=True)
            except:
                pass

        else:
            st.markdown("""<div style="border: 2px dashed #334155; border-radius: 12px; height: 300px; display: flex; align-items: center; justify-content: center; color: #64748b;">Esperando resultados...</div>""", unsafe_allow_html=True)

with tab_hist:
    st.markdown("### 📂 Registro Local de Pacientes")
    # FIX: Reemplazar experimental_rerun por rerun
    if st.button("🔄 Refrescar Tabla"):
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun() # Fallback para versiones viejas
            
    try:
        df_h = get_history(limit=50)
        if not df_h.empty:
            st.dataframe(df_h, use_container_width=True)
        else:
            st.info("La base de datos está vacía.")
    except:
        st.warning("No se pudo conectar a la base de datos.")

st.markdown("---")
st.caption("HematoAI CDSS v1.0 | Production Release | Matías Gacitúa Ruiz | MatiRCode")