import pandas as pd

from utilitarios import ver_tabela_nulos

# Carregando o arquivo
# Agent e company serão convertidos para string pois são IDs e não devem ser
# interpretados como valores numéricos
df = pd.read_csv(
    "hotel_bookings.csv",
    encoding="latin_1",
    dtype={
        "agent": str,
        "company": str,
    },
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
df["company"] = df["company"].fillna("-1")
print(f"Nulos restantes em company: {df['company'].isnull().sum()}", "\n")

# Os nulos de agent podem ser interpretados como reservas sem agência, que é uma
# opção válida. Preenchendo com ID -1
print("Os nulos de agent serão preenchidos por uma nova categoria -1.")
df["agent"] = df["agent"].fillna("-1")
print(f"Nulos restantes em agent: {df['agent'].isnull().sum()}", "\n")

# Os nulos de country e children serão removidos pois a falta desses valores
# não faz sentido
print("Removendo nulos de country e children.")
df = df.dropna(subset=["country", "children"])

print("Nulos restantes:")
ver_tabela_nulos(df)
