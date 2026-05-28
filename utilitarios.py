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
