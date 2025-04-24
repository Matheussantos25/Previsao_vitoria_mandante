import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# --- 0) Configurações da página: SEMPRE o primeiro comando st.* ---
st.set_page_config(page_title="Previsão Vitórias Mandante", layout="wide")

# --- 1) Carrega dados de referência para médias e encoders ---
@st.cache_data
def load_reference():
    df_ref = pd.read_csv("brasileiro_variaveis_historicas.csv")
    encoders = {}
    for col in ["Mandante","Visitante","Estádio","Cidade"]:
        le = LabelEncoder().fit(df_ref[col])
        encoders[col] = le
    climate_means = {
        "Temperatura (°C)": df_ref["Temperatura (°C)"].mean(),
        "Umidade (%)":    df_ref["Umidade (%)"].mean(),
        "Vento (km/h)":   df_ref["Vento (km/h)"].mean(),
    }
    return df_ref, encoders, climate_means

df_ref, encoders, climate_means = load_reference()

# --- 2) Carrega o modelo treinado ---
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

# --- 3) Layout ---
st.title("🏟️ Previsão de Vitória do Mandante")
tabs = st.tabs(["Previsão Individual", "Previsão em Lote", "Sobre o Modelo"])

# --- 3) Aba: Previsão Individual ---
with tabs[0]:
    st.header("Entrada Manual de Variáveis")

    # Parte categórica
    col1, col2 = st.columns(2)
    with col1:
        mandante = st.selectbox("Time Mandante",  encoders["Mandante"].classes_)
        visitante = st.selectbox("Time Visitante", encoders["Visitante"].classes_)
    with col2:
        estadio   = st.selectbox("Estádio",       encoders["Estádio"].classes_)
        cidade    = st.selectbox("Cidade",        encoders["Cidade"].classes_)

    st.markdown("---")
    # Clima opcional
    use_climate = st.checkbox("Informar variáveis de clima?", value=False)
    if use_climate:
        temp = st.number_input("Temperatura (°C)", step=0.1)
        umid = st.number_input("Umidade (%)",    step=0.1)
        vento= st.number_input("Vento (km/h)",   step=0.1)
    else:
        temp  = climate_means["Temperatura (°C)"]
        umid  = climate_means["Umidade (%)"]
        vento = climate_means["Vento (km/h)"]
        st.caption(f"> Usando média histórica: T={temp:.1f} °C, U={umid:.1f} %, V={vento:.1f} km/h")

    st.markdown("---")
    # Estatísticas dos últimos 3 jogos
    col3, col4 = st.columns(2)
    with col3:
        m3_saldo    = st.number_input("Mandante: Saldo últimos 3 jogos",   min_value=-50, max_value=50, step=1)
        m3_marcados = st.number_input("Mandante: Gols marcados (3 jogos)", min_value=0,   max_value=20, step=1)
        m3_sofridos = st.number_input("Mandante: Gols sofridos (3 jogos)", min_value=0,   max_value=20, step=1)
        pos_m       = st.number_input("Posição atual mandante",             min_value=1,   max_value=20, step=1)
    with col4:
        v3_saldo    = st.number_input("Visitante: Saldo últimos 3 jogos",   min_value=-50, max_value=50, step=1)
        v3_marcados = st.number_input("Visitante: Gols marcados (3 jogos)", min_value=0,   max_value=20, step=1)
        v3_sofridos = st.number_input("Visitante: Gols sofridos (3 jogos)", min_value=0,   max_value=20, step=1)
        pos_v       = st.number_input("Posição atual visitante",            min_value=1,   max_value=20, step=1)

    if st.button("🔮 Prever"):
        # Monta DataFrame de previsão
        row = {
            "Mandante": encoders["Mandante"].transform([mandante])[0],
            "Visitante":encoders["Visitante"].transform([visitante])[0],
            "Estádio":   encoders["Estádio"].transform([estadio])[0],
            "Cidade":    encoders["Cidade"].transform([cidade])[0],
            "Temperatura (°C)": temp,
            "Umidade (%)":      umid,
            "Vento (km/h)":     vento,
            "Mandante_3j_Saldo":    m3_saldo,
            "Mandante_3j_Marcados": m3_marcados,
            "Mandante_3j_Sofridos": m3_sofridos,
            "Mandante_Posicao":     pos_m,
            "Visitante_3j_Saldo":    v3_saldo,
            "Visitante_3j_Marcados": v3_marcados,
            "Visitante_3j_Sofridos": v3_sofridos,
            "Visitante_Posicao":     pos_v
        }
        X_new = pd.DataFrame([row])
        # pred e prob
        pred = model.predict(X_new)[0]
        prob = model.predict_proba(X_new)[0]
        label = {1:"🏆 Vitória", 0:"🤝 Empate", -1:"❌ Derrota"}[pred]

        st.subheader(f"📊 Resultado: **{label}**")
        st.bar_chart(pd.DataFrame({
            "Classe": ["Derrota(-1)","Empate(0)","Vitória(1)"],
            "Prob": prob
        }).set_index("Classe"))

# --- 4) Aba: Previsão em Lote ---
with tabs[1]:
    st.header("Upload de CSV para Previsão em Lote")
    uploaded = st.file_uploader("Envie um .csv com as mesmas colunas de entrada", type="csv")
    if uploaded:
        df_batch = pd.read_csv(uploaded)
        # aplica mesmas transformações
        for col in ["Mandante","Visitante","Estádio","Cidade"]:
            df_batch.loc[:, col] = encoders[col].transform(df_batch[col])
        # clima opcional: preenche NaNs com médias
        for c in ["Temperatura (°C)","Umidade (%)","Vento (km/h)"]:
            df_batch[c] = df_batch[c].fillna(climate_means[c])
        # previsões
        df_batch["Pred"]   = model.predict(df_batch[ df_batch.columns.intersection(row.keys()) ])
        probs = model.predict_proba(df_batch[ df_batch.columns.intersection(row.keys()) ])
        df_batch[["Prob_Derrota","Prob_Empate","Prob_Vitória"]] = list(probs)
        st.dataframe(df_batch)
        st.download_button(
            "📥 Baixar resultados",
            df_batch.to_csv(index=False),
            file_name="predicoes.csv",
            mime="text/csv"
        )

# --- 5) Aba: Sobre o Modelo ---
with tabs[2]:
    st.header("🧠 Sobre o Modelo")
    st.markdown("""
    **Algoritmo:** Random Forest (class_weight='balanced', max_depth=10, min_samples_leaf=5)  
    **Validação:** TimeSeriesSplit(n_splits=5)  
    **Métricas (CV)**  
    - Acurácia média: ~41–46%  
    - F1 Macro média: ~34–38%  
    - ROC AUC (OvR): ~55–60%  

    **Principais variáveis** (por importância):
    1. Time Visitante  
    2. Posição Visitante  
    3. Posição Mandante  
    4. Saldo últimos 3 jogos (mandante)  
    5. Saldo últimos 3 jogos (visitante)  

    **Objetivo:**  
    Estimar a probabilidade de vitória do mandante em partidas de futebol com base em desempenho recente, posição na tabela e variáveis de clima.

    **Limitações & Melhorias**  
    - Classes desbalanceadas: considerar técnicas avançadas de balanceamento (SMOTE).  
    - Incluir mais features: forma de 5 jogos, histórico de confrontos diretos, condições de viagem, etc.  
    - Atualizar modelo periodicamente com novos dados.
    """)
