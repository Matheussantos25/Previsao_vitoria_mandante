import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Configuração da página
st.set_page_config(
    page_title="Previsão de Vitória do Mandante",
    layout="wide"
)
warnings.filterwarnings("ignore")

# Paths dos artefatos
MODEL_PATH = 'best_model.pkl'
SCALER_PATH = 'scaler.pkl'
ENCODERS_PATH = 'encoders.pkl'

# Função de treinamento e salvamento dos melhores artefatos
@st.cache(allow_output_mutation=True)
def train_and_save(csv_path='brasileiro_variaveis_historicas.csv'):
    df = pd.read_csv(csv_path)
    df['Resultado'] = df['Placar'].apply(
        lambda p: 1 if (lambda m,v: m>v)(*map(int,p.split(' x '))) else 0 if (lambda m,v: m==v)(*map(int,p.split(' x '))) else -1
    )
    df.dropna(subset=['Resultado'], inplace=True)

    features = [
        'Mandante','Visitante','Estádio','Cidade',
        'Temperatura (°C)','Umidade (%)','Vento (km/h)',
        'Mandante_3j_Saldo','Mandante_3j_Marcados','Mandante_3j_Sofridos','Mandante_Posicao',
        'Visitante_3j_Saldo','Visitante_3j_Marcados','Visitante_3j_Sofridos','Visitante_Posicao'
    ]
    X = df[features].copy()
    y = df['Resultado']

    # Encoding
    cat_cols = ['Mandante','Visitante','Estádio','Cidade']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    # Scaling
    num_cols = [c for c in features if c not in cat_cols]
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Modelos
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'SVC': SVC(probability=True, random_state=42)
    }
    best_model, best_acc, best_name = None, -np.inf, None
    for name, m in models.items():
        m.fit(X_train, y_train)
        acc = accuracy_score(y_test, m.predict(X_test))
        if acc > best_acc:
            best_acc, best_model, best_name = acc, m, name

    # Salva artefatos
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(ENCODERS_PATH, 'wb') as f:
        pickle.dump(encoders, f)

    return best_name, best_acc

# Se artefatos não existem, treina e salva
if not os.path.exists(MODEL_PATH):
    name, acc = train_and_save()
    st.write(f"Treinado e salvo {name} com acurácia de {acc:.2%}")

# Carrega artefatos salvos
with open(MODEL_PATH,'rb') as f:
    model = pickle.load(f)
with open(SCALER_PATH,'rb') as f:
    scaler = pickle.load(f)
with open(ENCODERS_PATH,'rb') as f:
    encoders = pickle.load(f)

features = [
    'Mandante','Visitante','Estádio','Cidade',
    'Temperatura (°C)','Umidade (%)','Vento (km/h)',
    'Mandante_3j_Saldo','Mandante_3j_Marcados','Mandante_3j_Sofridos','Mandante_Posicao',
    'Visitante_3j_Saldo','Visitante_3j_Marcados','Visitante_3j_Sofridos','Visitante_Posicao'
]
classes = model.classes_

# Streamlit: abas
tabs = st.tabs(["Previsão Individual","Previsão em Lote","Sobre o Modelo"])

# Aba 1: individual
with tabs[0]:
    st.header("Previsão Individual de Vitória do Mandante")
    st.write("Selecione as informações abaixo:")
    home = st.selectbox("Time Mandante", encoders['Mandante'].classes_)
    away = st.selectbox("Time Visitante", encoders['Visitante'].classes_)
    estadio = st.selectbox("Estádio", encoders['Estádio'].classes_)
    cidade = st.selectbox("Cidade", encoders['Cidade'].classes_)

    use_weather = st.checkbox("Incluir clima (opcional)")
    clima = {}
    if use_weather:
        clima['Temperatura (°C)'] = st.number_input("Temperatura (°C)", value=25.0)
        clima['Umidade (%)'] = st.number_input("Umidade (%)", value=50.0)
        clima['Vento (km/h)'] = st.number_input("Vento (km/h)", value=10.0)

    gols = {
        'Mandante_3j_Saldo': st.number_input("Saldo de gols mandante (últimos 3)",0),
        'Mandante_3j_Marcados': st.number_input("Gols marcados mandante (últimos 3)",0),
        'Mandante_3j_Sofridos': st.number_input("Gols sofridos mandante (últimos 3)",0),
        'Mandante_Posicao': st.number_input("Posição mandante na tabela",1)
    }
    gols_v = {
        'Visitante_3j_Saldo': st.number_input("Saldo de gols visitante (últimos 3)",0),
        'Visitante_3j_Marcados': st.number_input("Gols marcados visitante (últimos 3)",0),
        'Visitante_3j_Sofridos': st.number_input("Gols sofridos visitante (últimos 3)",0),
        'Visitante_Posicao': st.number_input("Posição visitante na tabela",1)
    }

    if st.button("Prever"):
        data = {
            'Mandante': encoders['Mandante'].transform([home])[0],
            'Visitante': encoders['Visitante'].transform([away])[0],
            'Estádio': encoders['Estádio'].transform([estadio])[0],
            'Cidade': encoders['Cidade'].transform([cidade])[0]
        }
        data.update(clima)
        data.update(gols)
        data.update(gols_v)

        df_in = pd.DataFrame([data])
        num_cols = [c for c in features if c not in encoders]
        if clima:
            df_in[num_cols] = scaler.transform(df_in[num_cols])
        else:
            df_in[num_cols] = scaler.transform(df_in[num_cols])
        proba = model.predict_proba(df_in)[0]

        st.subheader("Probabilidades")
        dfp = pd.DataFrame({'Classe':classes,'Prob':[f"{p:.2%}" for p in proba]})
        st.table(dfp)
        st.markdown(f"**Probabilidade de vitória do mandante:** {proba[list(classes).index(1)]:.2%}")

# Aba 2: lote
with tabs[1]:
    st.header("Previsão em Lote via CSV")
    up = st.file_uploader("CSV com colunas de input", type='csv')
    if up:
        df = pd.read_csv(up)
        for col, le in encoders.items(): df[col] = le.transform(df[col])
        df[num_cols] = scaler.transform(df[num_cols])
        preds = model.predict(df)
        prox = model.predict_proba(df)
        df['Previsão'] = ['Vitória' if x==1 else 'Não Vitória' for x in preds]
        for i,cls in enumerate(classes): df[f'Prob_{cls}'] = prox[:,i]
        st.dataframe(df)
        st.bar_chart(df['Previsão'].value_counts())

# Aba 3: info
with tabs[2]:
    st.header("Sobre o Modelo")
    st.markdown(f"- **Modelo:** {os.path.basename(MODEL_PATH)}")
    st.markdown(f"- **Classes:** {list(classes)}")
    st.markdown(f"- **Features:** {features}")
    st.write("Este app carrega um modelo treinado que prevê vitória do mandante.")
    st.write("Limitações: depende de dados históricos; melhorias: incluir xG, SHAP para interpretabilidade.")
