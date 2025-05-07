# train_model.py

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# 1) Carrega dados e define target
df = pd.read_csv("brasileiro_variaveis_historicas.csv")
def resultado_mandante(placar):
    m, v = map(int, placar.split(" x "))
    return 1 if m > v else 0 if m == v else -1
df["Resultado_Num"] = df["Placar"].apply(resultado_mandante).map({-1:0, 0:1, 1:2})

# 2) Feature engineering: deltas
df["Delta_Posicao"]    = df["Mandante_Posicao"] - df["Visitante_Posicao"]
df["Delta_3j_Saldo"]   = df["Mandante_3j_Saldo"]  - df["Visitante_3j_Saldo"]
df["Delta_3j_Marcados"]= df["Mandante_3j_Marcados"] - df["Visitante_3j_Marcados"]
df["Delta_3j_Sofridos"]= df["Mandante_3j_Sofridos"] - df["Visitante_3j_Sofridos"]

cat_cols = ["Mandante", "Visitante"]
num_cols = [
    "Temperatura (°C)", "Umidade (%)", "Vento (km/h)",
    "Mandante_3j_Saldo", "Mandante_3j_Marcados", "Mandante_3j_Sofridos", "Mandante_Posicao",
    "Visitante_3j_Saldo", "Visitante_3j_Marcados", "Visitante_3j_Sofridos", "Visitante_Posicao",
    "Delta_Posicao", "Delta_3j_Saldo", "Delta_3j_Marcados", "Delta_3j_Sofridos"
]

X = df[cat_cols + num_cols]
y = df["Resultado_Num"]

# 3) Split estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# 4) Pré-processamento
preprocessor = ColumnTransformer([
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
    ("sca", StandardScaler(), num_cols),
])

# 5) Pipeline com SMOTE + XGBoost
pipeline = ImbPipeline([
    ("prep",  preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("clf",   XGBClassifier(
        objective="multi:softprob",
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42
    )),
])

# 6) Hiper-parâmetros
param_dist = {
    "clf__n_estimators":    [100, 200, 500],
    "clf__max_depth":       [3, 5, 7],
    "clf__learning_rate":   [0.01, 0.1, 0.2],
    "clf__subsample":       [0.6, 0.8, 1.0],
    "clf__colsample_bytree":[0.6, 0.8, 1.0],
    "clf__gamma":           [0, 1, 5],
    "clf__min_child_weight":[1, 3, 5],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=30,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# 7) Treina e busca melhores parâmetros
search.fit(X_train, y_train)

# 8) Salva o pipeline completo (pré-processamento + modelo)
best_pipeline = search.best_estimator_
joblib.dump(best_pipeline, "best_pipeline.pkl")

# 9) Avaliação final (opcional)
y_pred  = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)
print("Acurácia: ", accuracy_score(y_test, y_pred))
print("F1-Macro: ", f1_score(y_test, y_pred, average="macro"))
print("ROC AUC:  ", roc_auc_score(pd.get_dummies(y_test), y_proba, multi_class="ovr"))
print("\nReport:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
