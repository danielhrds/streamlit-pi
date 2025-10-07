import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("dataset_ocorrencias_delegacia_5.csv")
df["data_ocorrencia"] = pd.to_datetime(df["data_ocorrencia"])
df["ano_mes"] = df["data_ocorrencia"].dt.to_period("M").astype(str)

st.set_page_config(page_title="Dashboard de Gráficos", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    h1, h2, h3 { color: #1f77b4; }
    </style>
""", unsafe_allow_html=True)

st.title(" Dashboard Interativo com Streamlit")

option = st.sidebar.selectbox(
    "Navegar para:",
    [
        "Número de vítimas x Tempo",
        "Quantidade de vítimas x Bairro",
        "Mapa de calor (Roubos)",
        "Clusters",
        "Anomalias"
    ]
)

if option == "Número de vítimas x Tempo":
    df_homicidios = df[df["tipo_crime"].str.lower() == "homicídio"]
    serie = df_homicidios.groupby("ano_mes")["quantidade_vitimas"].sum().reset_index()

    fig = px.line(
        serie, 
        x="ano_mes", 
        y="quantidade_vitimas",
        labels={"quantidade_vitimas": "Nº de Vítimas", "ano_mes": "Ano-Mês"},
        title=" Evolução de Homicídios em Recife"
    )
    fig.update_traces(line=dict(width=3, color="#1f77b4"))
    st.plotly_chart(fig, use_container_width=True)

elif option == "Quantidade de vítimas x Bairro":
    df_homicidios = df[df["tipo_crime"].str.lower() == "homicídio"]
    ranking = df_homicidios.groupby("bairro")["quantidade_vitimas"].sum().reset_index()
    ranking = ranking.sort_values(by="quantidade_vitimas", ascending=False).head(10)

    fig = px.bar(
        ranking, 
        x="bairro", 
        y="quantidade_vitimas",
        text="quantidade_vitimas",
        color_discrete_sequence=["#1f77b4"],
        title=" Top 10 Bairros com Mais Vítimas de Homicídio"
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

elif option == "Mapa de calor (Roubos)":
    st.subheader(" Mapa de Calor - Roubos em Recife")

    meses_disponiveis = sorted(df["ano_mes"].unique())
    mes_selecionado = st.selectbox(" Selecionar mês:", options=meses_disponiveis, index=len(meses_disponiveis)-1)

    bairros = st.multiselect(
        "🏙️ Filtrar bairros:",
        options=sorted(df["bairro"].dropna().unique()),
        default=[]
    )
    
    df_filtered = df.copy()
    if bairros:
        df_filtered = df_filtered[df_filtered["bairro"].isin(bairros)]

    df_roubo = df_filtered[
        (df_filtered["tipo_crime"].str.lower() == "roubo") &
        (df_filtered["ano_mes"] == mes_selecionado)
    ].copy()

    if df_roubo.empty:
        st.warning(" Nenhum dado de roubo encontrado para o mês selecionado.")
    else:
        df_expanded = df_roubo.loc[df_roubo.index.repeat(df_roubo["quantidade_vitimas"])].reset_index(drop=True)

        fig = px.density_mapbox(
            df_expanded,
            lat="latitude",
            lon="longitude",
            radius=15,
            center={"lat": -8.05, "lon": -34.9},
            zoom=11,
            title=f" Mapa de Calor - Roubos em Recife ({mes_selecionado})",
            mapbox_style="carto-positron"
        )

        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
         Este mapa mostra a concentração de **roubos** por bairro em Recife no mês de **{mes_selecionado}**, com intensidade baseada no número de vítimas registradas.
        """)

elif option == "Clusters":
    st.header(" Agrupamento de Bairros por Perfis de Roubos")

    df_roubo = df[df["tipo_crime"].str.lower() == "roubo"].copy()
    df_cluster = df_roubo.groupby("bairro").agg({
        "quantidade_vitimas": "sum",
        "latitude": "mean",
        "longitude": "mean"
    }).dropna()

    if df_cluster.empty:
        st.warning(" Nenhum dado de roubo encontrado no dataset.")
    else:
        X = df_cluster[["quantidade_vitimas", "latitude", "longitude"]]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_clusters = st.slider("Número de clusters:", 2, 6, 4)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df_cluster["cluster"] = kmeans.fit_predict(X_scaled)
        df_cluster["cluster"] = df_cluster["cluster"].astype(str)

        st.write("###  Descrição dos clusters (baseado apenas em roubos)")
        for c in sorted(df_cluster["cluster"].unique()):
            qtd = df_cluster[df_cluster["cluster"] == c].shape[0]
            media = df_cluster[df_cluster["cluster"] == c]["quantidade_vitimas"].mean()
            st.markdown(f"- **Cluster {c}** → {qtd} bairros, média de vítimas de roubo: {media:.2f}")

        fig = px.scatter_mapbox(
            df_cluster,
            lat="latitude",
            lon="longitude",
            color="cluster",
            size="quantidade_vitimas",
            hover_name=df_cluster.index,
            zoom=11,
            title="Clusters de Bairros por Ocorrências de Roubo",
            mapbox_style="carto-positron",
            color_discrete_sequence=["#FF7F00", "#FF0000", "#008000", "#001F3F", "#800080", "#FFD700"]
        )
        st.plotly_chart(fig, use_container_width=True)
elif option == "Anomalias":
    st.header(" Detecção de Anomalias no Número de Vítimas de Roubo")

    df_roubo = df[df["tipo_crime"].str.lower() == "roubo"].copy()
    df_ano = df_roubo.groupby("ano_mes")["quantidade_vitimas"].sum().reset_index()

    if df_ano.empty:
        st.warning(" Nenhum dado de roubo disponível para detecção de anomalias.")
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(df_ano[["quantidade_vitimas"]])

        iso = IsolationForest(contamination=0.1, random_state=42)
        df_ano["anomaly"] = iso.fit_predict(X)
        df_ano["anomaly_label"] = df_ano["anomaly"].map({1: "Normal", -1: "Anômalo"})

        fig = px.scatter(
            df_ano,
            x="ano_mes",
            y="quantidade_vitimas",
            color="anomaly_label",
            title=" Anomalias no Número de Vítimas de Roubo ao Longo do Tempo",
            labels={"quantidade_vitimas": "Número de Vítimas de Roubo", "ano_mes": "Ano-Mês"},
            color_discrete_map={"Normal": "#1f77b4", "Anômalo": "#FF0000"}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
         Pontos marcados como **Anômalo** representam meses com valores de **roubos** muito fora do padrão esperado.
        """)

st.markdown("---")
