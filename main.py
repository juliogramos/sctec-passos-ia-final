import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utilitarios import outlier_calc, ver_tabela_nulos

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
plt.show()

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
# Por enquanto, não vou remover nenhum outlier, pois eles podem ter correlações
# importantes para a modelagem preditiva.
# Irei checar essas correlações no próximo passo.
