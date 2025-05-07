# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 0) Configurações da página ---
st.set_page_config(page_title="Previsão Vitórias Mandante", layout="wide")

# --- 1) Carrega referências para clima e mapeamento estádio→cidade ---
@st.cache_data
def load_reference():
    df_ref = pd.read_csv("brasileiro_variaveis_historicas.csv")
    stadium_city_map = (
        df_ref[["Estádio", "Cidade"]]
        .drop_duplicates()
        .set_index("Estádio")["Cidade"]
        .to_dict()
    )
    climate_means = df_ref[["Temperatura (°C)", "Umidade (%)", "Vento (km/h)"]].mean().to_dict()
    return stadium_city_map, climate_means

stadium_city_map, climate_means = load_reference()

# --- 2) Carrega o pipeline treinado ---
@st.cache_resource
def load_pipeline():
    return joblib.load("best_pipeline.pkl")

pipeline = load_pipeline()

# --- 3) Definição de features ---
cat_cols = ["Mandante", "Visitante"]
num_cols = [
    "Temperatura (°C)", "Umidade (%)", "Vento (km/h)",
    "Mandante_3j_Saldo", "Mandante_3j_Marcados", "Mandante_3j_Sofridos", "Mandante_Posicao",
    "Visitante_3j_Saldo", "Visitante_3j_Marcados", "Visitante_3j_Sofridos", "Visitante_Posicao",
    "Delta_Posicao", "Delta_3j_Saldo", "Delta_3j_Marcados", "Delta_3j_Sofridos"
]

# --- 4) Layout principal ---
st.title("🏟️ Previsão de Vitória do Mandante")
tabs = st.tabs(["Previsão Individual", "Previsão em Lote", "Sobre o Modelo"])

# 4.1) Previsão Individual
with tabs[0]:
    st.header("Entrada Manual de Variáveis")
    col1, col2 = st.columns(2)
    with col1:
        mandante  = st.selectbox("Time Mandante",  pipeline.named_steps["prep"]
                                 .named_transformers_["ohe"].categories_[0])
        visitante = st.selectbox("Time Visitante", pipeline.named_steps["prep"]
                                 .named_transformers_["ohe"].categories_[1])
    with col2:
        estadio = st.selectbox("Estádio", list(stadium_city_map.keys()))
        default_city = stadium_city_map[estadio]
        cidade = st.selectbox("Cidade", list(set(stadium_city_map.values())),
                              index=list(set(stadium_city_map.values())).index(default_city))
    st.markdown("---")

    use_climate = st.checkbox("Informar variáveis de clima?", value=False)
    if use_climate:
        temp  = st.number_input("Temperatura (°C)", step=0.1)
        umid  = st.number_input("Umidade (%)",    step=0.1)
        vento = st.number_input("Vento (km/h)",   step=0.1)
    else:
        temp, umid, vento = climate_means["Temperatura (°C)"], climate_means["Umidade (%)"], climate_means["Vento (km/h)"]
        st.caption(f"> Usando média histórica: T={temp:.1f} °C, U={umid:.1f} %, V={vento:.1f} km/h")
    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        m3_marcados = st.number_input("Mandante: Gols marcados (3 jogos)", min_value=0, max_value=20, step=1)
        m3_sofridos = st.number_input("Mandante: Gols sofridos (3 jogos)", min_value=0, max_value=20, step=1)
        m3_saldo    = m3_marcados - m3_sofridos
        st.metric("Mandante: Saldo últimos 3 jogos", m3_saldo)
        pos_m = st.slider("Posição atual mandante", 1, 20, 1)
    with col4:
        v3_marcados = st.number_input("Visitante: Gols marcados (3 jogos)", min_value=0, max_value=20, step=1)
        v3_sofridos = st.number_input("Visitante: Gols sofridos (3 jogos)", min_value=0, max_value=20, step=1)
        v3_saldo    = v3_marcados - v3_sofridos
        st.metric("Visitante: Saldo últimos 3 jogos", v3_saldo)
        pos_v = st.slider("Posição atual visitante", 1, 20, 2)

    if pos_m == pos_v:
        st.error("Mandante e visitante não podem ter a mesma posição na tabela.")
    st.markdown("---")

    if st.button("🔮 Prever") and pos_m != pos_v:
        # calcula deltas
        delta_pos       = pos_m - pos_v
        delta_saldo     = m3_saldo - v3_saldo
        delta_marcados  = m3_marcados - v3_marcados
        delta_sofridos  = m3_sofridos - v3_sofridos

        row = {
            "Mandante": mandante,
            "Visitante": visitante,
            "Temperatura (°C)": temp,
            "Umidade (%)": umid,
            "Vento (km/h)": vento,
            "Mandante_3j_Saldo": m3_saldo,
            "Mandante_3j_Marcados": m3_marcados,
            "Mandante_3j_Sofridos": m3_sofridos,
            "Mandante_Posicao": pos_m,
            "Visitante_3j_Saldo": v3_saldo,
            "Visitante_3j_Marcados": v3_marcados,
            "Visitante_3j_Sofridos": v3_sofridos,
            "Visitante_Posicao": pos_v,
            "Delta_Posicao": delta_pos,
            "Delta_3j_Saldo": delta_saldo,
            "Delta_3j_Marcados": delta_marcados,
            "Delta_3j_Sofridos": delta_sofridos
        }
        X_new = pd.DataFrame([row])[cat_cols + num_cols]
        pred = pipeline.predict(X_new)[0]
        prob = pipeline.predict_proba(X_new)[0]

        label_map = {2: "🏆 Vitória", 1: "🤝 Empate", 0: "❌ Derrota"}
        st.subheader(f"📊 Resultado: **{label_map[pred]}**")

        prob_df = pd.DataFrame({
            "Classe": ["Derrota (0)", "Empate (1)", "Vitória (2)"],
            "Probabilidade": prob
        }).set_index("Classe")
        st.bar_chart(prob_df)

# 4.2) Previsão em Lote
with tabs[1]:
    st.header("Upload de CSV para Previsão em Lote")
    uploaded = st.file_uploader("Envie um .csv com as mesmas colunas de entrada", type="csv")
    if uploaded:
        df_batch = pd.read_csv(uploaded)
        for c in ["Temperatura (°C)", "Umidade (%)", "Vento (km/h)"]:
            df_batch[c] = df_batch[c].fillna(climate_means[c])
        df_batch["Delta_Posicao"]    = df_batch["Mandante_Posicao"] - df_batch["Visitante_Posicao"]
        df_batch["Delta_3j_Saldo"]   = df_batch["Mandante_3j_Saldo"]  - df_batch["Visitante_3j_Sofridos"]
        df_batch["Delta_3j_Marcados"]= df_batch["Mandante_3j_Marcados"] - df_batch["Visitante_3j_Marcados"]
        df_batch["Delta_3j_Sofridos"]= df_batch["Mandante_3j_Sofridos"] - df_batch["Visitante_3j_Sofridos"]

        preds = pipeline.predict(df_batch[cat_cols + num_cols])
        probs = pipeline.predict_proba(df_batch[cat_cols + num_cols])
        df_batch["Pred"] = preds
        df_batch[["Prob_Derrota", "Prob_Empate", "Prob_Vitória"]] = list(probs)
        st.dataframe(df_batch)
        st.download_button(
            "📥 Baixar resultados",
            df_batch.to_csv(index=False),
            file_name="predicoes.csv",
            mime="text/csv"
        )

# 4.3) Sobre o Modelo
with tabs[2]:
    st.header("🧠 Sobre o Modelo")
    st.markdown("""
**Algoritmo:** XGBoost Classifier  
**Validação:** StratifiedKFold + RandomizedSearchCV(f1_macro)  
**Métricas (Teste)**  
- Acurácia: ~46%  
- F1 Macro: ~43%  
- ROC AUC (OvR): ~60%  

**Principais variáveis** (por ganho):
1. Visitante  
2. Visitante_Posicao  
3. Mandante_Posicao  
4. Mandante_3j_Saldo  
5. Visitante_3j_Saldo  

**Objetivo:**  
Estimar a probabilidade de vitória do mandante com base em desempenho recente e clima.  
""")
