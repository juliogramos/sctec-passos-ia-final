import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utilitarios import outlier_calc, plot_params_and_show, ver_tabela_nulos

# Carregando o arquivo
df = pd.read_csv(
    "hotel_bookings.csv",
    encoding="latin_1",
)

# Visualização inicial dos dados
print("VISUALIZAÇÃO INICIAL DOS DADOS:")
print(df.head(), "\n")

# Visualização dos tipos de dados
print("VISUALIZAÇÃO DOS TIPOS DE DADOS:")
print(df.info(), "\n")

# Tratamento de duplicados
print("TRATAMENTO DE DUPLICADOS:")
duplicados = df.duplicated().sum()
print(
    f"Duplicados encontrados: {duplicados}, {duplicados / len(df) * 100:.2f}% do dataset"
)

# 1/4 do dataset me parece muito para remover
# Como não há ID do cliente ou da reserva, não dá para saber se realmente são
# reservas duplicadas ou se são reservas únicas com os mesmos parâmetros.
# Por essas razões, decidi não remover os duplicados.

print("Duplicados não serão removidos.")
print("1/4 do dataset me parece muito para remover.")
print(
    "Como não há ID do cliente ou da reserva, não dá para saber se realmente são reservas duplicadas ou se são reservas únicas com os mesmos parâmetros.",
    "\n",
)

# Tratamento de nulos
print("VERIFICAÇÃO DE NULOS:")
print("Nulos encontrados:")
ver_tabela_nulos(df)


print("TRATAMENTO DE NULOS:")
# Os nulos de company são quase todo o dataset e serão preenchidos com uma nova
# categoria, ID -1
print("Os nulos de company serão preenchidos por uma nova categoria -1.")
df["company"] = df["company"].fillna(-1)
print(f"Nulos restantes em company: {df['company'].isnull().sum()}", "\n")

# Os nulos de agent podem ser interpretados como reservas sem agência, que é uma
# opção válida. Preenchendo com ID -1
print("Os nulos de agent serão preenchidos por uma nova categoria -1.")
df["agent"] = df["agent"].fillna(-1)
print(f"Nulos restantes em agent: {df['agent'].isnull().sum()}", "\n")

# Os nulos de country e children serão removidos pois a falta desses valores
# não faz sentido
print("Removendo nulos de country e children.")
df = df.dropna(subset=["country", "children"])

print("Nulos restantes:")
ver_tabela_nulos(df)

# Distribuição de features
numeric_features = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

categoric_features = [
    "hotel",
    "is_canceled",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "agent",
    "company",
    "customer_type",
    "reservation_status",
]

# Verificação de outliers por box plots
fig, axs = plt.subplots(nrows=2, ncols=7, figsize=(25, 25))
axs = axs.flatten()
fig.delaxes(axs[13])

for i, col in enumerate(numeric_features):
    sns.boxplot(y=df[col], ax=axs[i], color="skyblue")
    axs[i].set_title(f"Box plot de {col}")
    axs[i].set_xlabel("")

plt.tight_layout()
# plt.show()

# À primeira vista, apenas o total_of_special_requests é um bom candidato para
# a remoção de outliers

# Verificação de outliers por IQR
print("IDENTIFICANDO OUTLIERS POR IQR:")
outliers_list = []
for col in numeric_features:
    outliers, percentual = outlier_calc(df, col)
    outliers_list.append([outliers, percentual])
outliers_df = pd.DataFrame(
    outliers_list, index=numeric_features, columns=["Total outliers", "Percentual (%)"]
)
outliers_df = outliers_df.sort_values(by="Total outliers", ascending=False)
print(outliers_df)
print(
    f"Total de outliers: {outliers_df['Total outliers'].sum()}, {outliers_df['Percentual (%)'].sum():.2f} do dataset.",
    "\n",
)

# Quais remover?
# Por enquanto, não vou remover nenhum outlier, pois eles podem ter relações
# importantes para a modelagem preditiva.
# Irei checar essas relações no próximo passo.

# Análise exploratória
print("ANÁLISE EXPLORATÓRIA:")

# Matriz de correlações
df_corr = df[numeric_features].copy()
correlation_matrix = df_corr.corr()
print("Matriz de correlação:")
print(correlation_matrix.round(4), "\n")

plt.figure(figsize=(10, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    cbar=True,
)
plt.title("Matriz de correlação de features numéricas")
# plt.show()

# Correlações interessantes encontradas:
# 1. stays_in_weekend_nights com stays_in_week_nights
# 2. children e adults com adr

# Relações a serem examinadas:
# 1. cancelamento por lead time
df_lead_time = df.groupby("lead_time")
df_lead_time.size().plot(kind="line")
plot_params_and_show("Distribuição de lead times", "Lead time", "Reservas", 45)
print("Top 10 Lead Times:")
print(df_lead_time.size().nlargest(10), "\n")

# Bons candidatos para outliers
print(
    f"Lead time maior que 450: {len(df[df['lead_time'] > 450])}, {len(df[df['lead_time'] > 450]) / len(df) * 100:.2f}% do dataset"
)

# Removendo esses outliers
print("Removendo lead time maior que 450")
index = df[df["lead_time"] > 450].index
df = df.drop(index)
print(
    f"Lead time maior que 450: {len(df[df['lead_time'] > 450])}, {len(df[df['lead_time'] > 450]) / len(df) * 100:.2f}% do dataset"
)

# Repetindo o gráfico inicial de lead time
df_lead_time = df.groupby("lead_time")
df_lead_time.size().plot(kind="line", figsize=(20, 10))
plot_params_and_show("Distribuição de lead times", "Lead time", "Reservas", 45)

# Como o is_canceled já está sendo codificado como 0 e 1, é possível descobrir o
# número de cancelados com uma simples soma, sem precisar de um filtro
df_lead_time["is_canceled"].sum().plot(kind="line")
plot_params_and_show("Cancelamentos por lead time", "Lead time", "Cancelamentos", 45)

# 2. cancelamento por tipo de hotel
df_tipo_hotel = df.groupby("hotel")
df_tipo_hotel.size().plot(kind="pie")
plot_params_and_show("Distribuição de tipos de hotel", "Tipo do hotel", "Reservas", 45)

df_tipo_hotel["is_canceled"].sum().plot(kind="bar")
plot_params_and_show(
    "Cancelamentos por tipo do hotel", "Tipo do hotel", "Cancelamentos", 45
)

# 3. cancelamento por mês
# Gambiarra para mostrar os meses em ordem no gráfico
# Essa coluna nova não vai para as features de ML
num_para_mes_map = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

mes_para_num_map = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

df["MY_mes_numero"] = df["arrival_date_month"].map(mes_para_num_map)

df_meses = df.sort_values(["MY_mes_numero"], ascending=True).groupby("MY_mes_numero")
df_meses.size().plot(kind="bar").set_xticklabels(list(num_para_mes_map.values()))
plot_params_and_show("Distribuição de meses", "Mês", "Reservas", 45)

df_meses["is_canceled"].sum().plot(kind="bar").set_xticklabels(
    list(num_para_mes_map.values())
)
plot_params_and_show("Cancelamentos por mês", "Tipo do hotel", "Cancelamentos", 45)

# 4. cancelamento por adr
df_adr = df.groupby("adr")
df_adr.size().plot(kind="line", figsize=(20, 10))
plot_params_and_show("Distribuição de ADR", "ADR", "Reservas", 45)
print("Top 10 ADRs:")
print(df_adr.size().nlargest(10), "\n")

df_adr_ate_mil = df[df["adr"] <= 1000.0].groupby("adr")
df_adr_ate_mil.size().plot(kind="line", figsize=(20, 10))
plot_params_and_show("Distribuição de ADR até 1000", "ADR", "Reservas", 45)

print(f"ADRs negativos: {len(df[df['adr'] < 0])}")
print(f"ADRs nulos: {len(df[df['adr'] == 0])}")

# O adr negativo me parece um erro então irei remover.
# Mas adr nulo pode ser legítimo, como um hotel com todas as despesas pagas.
print("Removendo ADR negativo")
index = df[df["adr"] < 0].index
df = df.drop(index)
print(f"ADRs negativos: {len(df[df['adr'] < 0])}")

# Repetindo o gráfico anterior após isso
df_adr_ate_mil = df[df["adr"] <= 1000.0].groupby("adr")
df_adr_ate_mil.size().plot(kind="line", figsize=(20, 10))
plot_params_and_show(
    "Distribuição de ADR até 1000 após limpeza de negativos", "ADR", "Reservas", 45
)

print(
    f"ADRs maiores que 300: {len(df[df['adr'] > 300])}, {len(df[df['adr'] > 300]) / len(df) * 100:.2f}% do dataset"
)

# Agora sim, irei remover esses outliers.
print("Removendo outliers ADR")
index = df[df["adr"] > 300].index
df = df.drop(index)
print(
    f"ADRs maiores que 300: {len(df[df['adr'] > 300])}, {len(df[df['adr'] > 300]) / len(df) * 100:.2f}% do dataset"
)

# Redesenhando o gráfico inicial de ADR
df_adr = df.groupby("adr")
df_adr.size().plot(kind="line", figsize=(20, 10))
plot_params_and_show("Distribuição de ADR sem outliers", "ADR", "Reservas", 45)

print("Top 10 ADRs com mais reservas:")
print(df_adr.size().nlargest(10))

df_adr["is_canceled"].sum().plot(kind="line", figsize=(20, 10))
plot_params_and_show("Cancelamentos por ADR", "ADR", "Cancelamentos", 45)

# Não consegui ajustar o gráfico para ser mais detalhado, então:
print("10 ADRs com mais cancelamentos:")
print(df_adr["is_canceled"].sum().nlargest(10), "\n")

# 5. cancelamento por país
df_paises = df.groupby("country")
df_paises.size().nlargest(10).plot(kind="bar")
plot_params_and_show("Top 10 Países com mais Reservas", "País", "Reservas", 45)

df_paises["is_canceled"].sum().nlargest(10).plot(kind="bar")
plot_params_and_show(
    "Top 10 Países com mais Cancelamentos", "País", "Cancelamentos", 45
)

# 6. cancelamento por is_repeated_guest
df_repeated_guest = df.groupby("is_repeated_guest")
df_repeated_guest.size().plot(kind="pie", labels=["Novos", "Repetidos"], figsize=(8, 8))
plot_params_and_show("Distribuição de clientes repetidos", "Clientes", "Reservas", 45)

df_repeated_guest["is_canceled"].sum().plot(kind="bar").set_xticklabels(
    ["Novos", "Repetidos"]
)
plot_params_and_show(
    "Cancelamentos por tipo de cliente", "Tipo do hotel", "Cancelamentos", 45
)

# 7. cancelamento por previous_cancelations
df_previous_cancelations = df.groupby("previous_cancellations")
df_previous_cancelations.size().plot(kind="bar")
plot_params_and_show(
    "Distribuição de cancelamentos prévios", "Cancelamentos prévios", "Reservas", 45
)

df_previous_cancelations["is_canceled"].sum().plot(kind="bar")
plot_params_and_show(
    "Cancelamento por cancelamentos prévios",
    "Cancelamentos prévios",
    "Cancelamentos",
    45,
)

# Interessante em teoria, mas é óbvio que os que tem cancelamentos prévios tem
# mais cancelamentos.

# Resolvi não analisar o previous_bookings_not_changed já que é um número
# complementar ao previous_cancellations.

# 8. cancelamento por booking_changes
df_booking_changes = df.groupby("booking_changes")
df_booking_changes.size().plot(kind="bar")
plot_params_and_show(
    "Distribuição de mudanças na reserva", "Mudanças na reserva", "Reservas", 45
)

df_booking_changes["is_canceled"].sum().plot(kind="bar")
plot_params_and_show(
    "Cancelamento por mudanças na reserva",
    "Mudanças na reserva",
    "Cancelamentos",
    45,
)

# Bem interessante e com sentido.

# 9. cancelamento por days_in_waiting_list
df_waiting_list = df.groupby("days_in_waiting_list")
df_waiting_list.size().plot(kind="line", figsize=(20, 10))
plot_params_and_show(
    "Distribuição de dias na lista de espera", "Dias na lista de espera", "Reservas", 45
)
print("10 nº de dias na lista de espera mais comuns:")
print(df_waiting_list.size().nlargest(10), "\n")

df_waiting_list["is_canceled"].sum().plot(kind="line", figsize=(20, 10))
plot_params_and_show(
    "Cancelamento por dias na lista de espera",
    "Dias na lista de espera",
    "Cancelamentos",
    45,
)
print("10 nº de dias na lista de espera com mais cancelamentos:")
print(df_waiting_list["is_canceled"].sum().nlargest(10), "\n")

# Interessante, a maioria dos cancelamentos é imediato

# 10. cancelamento por total_of special requests
df_special_requests = df.groupby("total_of_special_requests")
df_special_requests.size().plot(kind="bar")
plot_params_and_show(
    "Distribuição de total de pedidos especiais",
    "Total de pedidos especiais",
    "Reservas",
    45,
)

df_special_requests["is_canceled"].sum().plot(kind="bar")
plot_params_and_show(
    "Cancelamento por total de pedidos especiais",
    "Total de pedidos especiais",
    "Cancelamentos",
    45,
)

# Modelagem Preditiva
# Feature Engineering
# Todas as features criadas por mim terão o prefixo MY
print("CRIANDO NOVAS FEATURES (prefixo MY)")

print("HÓSPEDES TOTAIS")
df["MY_total_guests"] = df["adults"] + df["children"] + df["babies"]
print(df["MY_total_guests"], "\n")

print("ESTADIA TOTAL")
df["MY_total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
print(df["MY_total_nights"], "\n")

print("ADICIONANDO FEATURES À LISTA", "\n")
numeric_features.append("MY_total_guests")
numeric_features.append("MY_total_nights")

# Removendo Features
print("REMOVENDO FEATURES")

# Removendo o Target
categoric_features.remove("is_canceled")
categoric_features.remove("reservation_status")
print("TARGET (is_canceled) removido")

# Removendo o ano
# Não acho que o ano seja um bom valor para incluir, primeiro por que supostamente
# vamos usar o modelo para prever valores futuros, e as variáveis que fazem que
# um ano seja diferente do outro são inúmeras e fora do escopo dos dados.
categoric_features.remove("arrival_date_year")
print("Ano (arrival_date_year) removido")

# Removendo o dia do mês
# Não acho que seja útil, acho que o dia da semana seria mais útil mas não é
# um dado incluído no dataset
categoric_features.remove("arrival_date_day_of_month")
print("Dia do mês (arrival_date_day_of_month) removido", "\n")


print("SEPARANDO FEATURES PARA O MODELO")
FEATURES = numeric_features + categoric_features
TARGET = "is_canceled"
print(f"FEATURES: {FEATURES}")
print(f"TARGET: {TARGET}", "\n")

print("CRIANDO CÓPIA DO DATAFRAME")
df_ml = df[FEATURES + [TARGET]].copy()

print("VALIDANDO NOVO DATAFRAME")
print(df_ml.info())
print("NOVO DATAFRAME É VÁLIDO", "\n")

# Separação de dados de treino e teste
print("SEPARAÇÃO DE DADOS DE TREINO E TESTE")
X = df_ml.drop(TARGET, axis=1)
y = df_ml[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5, stratify=y
)
print("DADOS SEPARADOS ENTRE TREINO E TESTE")

# Pré-processamento dos dados
print("PRÉ-PROCESSAMENTO DOS DADOS")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categoric_features,
        ),
    ],
    remainder="drop",
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()

print("PRÉ-PROCESSAMENTO CONCLUÍDO")
print(f"Shape de X_train_processed: {X_train_processed.shape}")
print(f"Shape de X_test_processed: {X_test_processed.shape}")
print(f"Total de novas colunas: {len(feature_names)}", "\n")

# Reconstituindo Dataframe
print("RECONSTITUINDO DATAFRAME")
X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)  # type: ignore
X_train_df.index = X_train.index
print("Datarframe reconstituído")
print(X_train_df.head(), "\n")

# Treinando modelos (supervisionados)
print("TREINANDO MODELOS (supervisionados)", "\n")
metricas = {}

# Modelo 1: Regressão Logística
print("Modelo 1: Regressão Logística")
model_lr = LogisticRegression(random_state=5, solver="liblinear", max_iter=1000)
model_lr.fit(X_train_processed, y_train)
y_pred_lr = model_lr.predict(X_test_processed)
y_proba_lr = model_lr.predict_proba(X_test_processed)[:, 1]
metricas["Regressão Logística"] = {
    "Accuracy": accuracy_score(y_test, y_pred_lr),
    "AUC": roc_auc_score(y_test, y_proba_lr),
}
print("Treinamento de regressão logística concluído", "\n")

# Modelo 2: Random Forest
print("Modelo 2: Random Forest")
model_rf = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=5, n_jobs=1
)
model_rf.fit(X_train_processed, y_train)
y_pred_rf = model_rf.predict(X_test_processed)
y_proba_rf = model_rf.predict_proba(X_test_processed)[:, 1]

metricas["Random Forest"] = {
    "Accuracy": accuracy_score(y_test, y_pred_rf),
    "AUC": roc_auc_score(y_test, y_proba_rf),
}
print("Treinamento de Random Forest concluído", "\n")

# Resultados
print("RESULTADOS")

# Regressão Logística
print("Resultado: Regressão Logística")
print(
    classification_report(
        y_test, y_pred_lr, target_names=["Completed (0)", "Cancelled (1)"]
    ),
    "\n",
)

plt.figure(figsize=(6, 5))
sns.heatmap(
    confusion_matrix(y_test, y_pred_lr),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Previsto: Não Cancela (0)", "Previsto: Cancela (1)"],
    yticklabels=["Real: Não Cancela (0)", "Real: Cancela (1)"],
)
plt.title("Matriz de Confusão (Regressão Logística)")
plt.ylabel("Valores Reais")
plt.xlabel("Valores Previstos")
plt.show()

# Random Forest
print("Resultado: Random Forest")
print(
    classification_report(
        y_test, y_pred_rf, target_names=["Completed (0)", "Cancelled (1)"]
    ),
    "\n",
)

plt.figure(figsize=(6, 5))
sns.heatmap(
    confusion_matrix(y_test, y_pred_rf),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Previsto: Não Cancela (0)", "Previsto: Cancela (1)"],
    yticklabels=["Real: Não Cancela (0)", "Real: Cancela (1)"],
)
plt.title("Matriz de Confusão (Random Forest)")
plt.ylabel("Valores Reais")
plt.xlabel("Valores Previstos")
plt.show()

# Comparação entre modelos
print("COMPARAÇÃO ENTRE MODELOS")
df_comparacao = pd.DataFrame(metricas).T
print(df_comparacao)

print(f"\nMelhor Performance (AUC): {df_comparacao['AUC'].idxmax()}")
print(f"Melhor AUC: {df_comparacao['AUC'].max():.4f}")

# Curva ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
auc_lr_val = roc_auc_score(y_test, y_proba_lr)
auc_rf_val = roc_auc_score(y_test, y_proba_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f"Regressão Logística (AUC = {auc_lr_val:.2f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf_val:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Chute Aleatório (AUC = 0.50)")  # Linha de base
plt.xlabel("Taxa de Falsos Positivos (FPR)")
plt.ylabel("Taxa de Verdadeiros Positivos (TPR) / Recall")
plt.title("Curva ROC para Previsão de Cancelamento")
plt.legend()
plt.grid(True)
plt.show()
