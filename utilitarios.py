from pandas import DataFrame


def ver_tabela_nulos(df: DataFrame):
    nulos_porcento = df.isnull().sum() / len(df) * 100
    nulos_df = DataFrame(
        {"Total nulos": df.isnull().sum(), "Percentual (%)": nulos_porcento.round(2)}
    )
    nulos_df = nulos_df[nulos_df["Total nulos"] > 0].sort_values(
        by="Total nulos", ascending=False
    )
    print(nulos_df, "\n")


def outlier_calc(df: DataFrame, coluna: str):
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = df[(df[coluna] < limite_inferior) | (df[coluna] > limite_superior)]
    return len(outliers), len(outliers) / len(df) * 100
