import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

df = pd.read_csv("dataset_ocorrencias_delegacia_5.csv")

df["data_ocorrencia"] = pd.to_datetime(df["data_ocorrencia"])

df["ano_mes"] = df["data_ocorrencia"].dt.to_period("M").astype(str)
st.set_page_config(
    page_title="Dashboard de Gráficos",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stSelectbox label {
        font-weight: bold;
        color: #fff;
    }
    h1, h2, h3 {
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Dashboard Interativo com Streamlit")

data = pd.DataFrame({
    "x": np.linspace(0, 10, 100),
    "Seno": np.sin(np.linspace(0, 10, 100)),
    "Cosseno": np.cos(np.linspace(0, 10, 100))
})

st.subheader("Escolha o tipo de gráfico")
option = st.selectbox(
    "Gráficos",
    ["Número de vítimas x Tempo", "Quantidade de vítimas x Bairro", "Mapa de calor"]
)

if option == "Número de vítimas x Tempo":
    df_homicidios = df[df["tipo_crime"].str.lower() == "homicídio"]
    serie = df_homicidios.groupby("ano_mes")["quantidade_vitimas"].sum().reset_index()

    fig = px.line(serie, x="ano_mes", y="quantidade_vitimas",
                  labels={"quantidade_vitimas": "Nº de Vítimas", "ano_mes": "Ano-Mês"},
                  title="Evolução de Homicídios em Recife")
    fig.update_traces(line=dict(width=3, color="#1f77b4"))
    st.plotly_chart(fig, use_container_width=True)

elif option == "Quantidade de vítimas x Bairro":
    df_homicidios = df[df["tipo_crime"].str.lower() == "homicídio"]
    ranking = df_homicidios.groupby("bairro")["quantidade_vitimas"].sum().reset_index()
    ranking = ranking.sort_values(by="quantidade_vitimas", ascending=False).head(10)

    fig = px.bar(ranking, x="bairro", y="quantidade_vitimas",
                 text="quantidade_vitimas",
                 color_discrete_sequence=["#1f77b4"],
                 title="Top 10 Bairros com Mais Vítimas de Homicídio")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

elif option == "Mapa de calor":
  st.subheader("Escolha o tipo de crime")

  bairros = st.multiselect(
    "Filtrar bairros:",
    options=sorted(df["bairro"].dropna().unique()),
    default=[]
  )
  
  df_filtered = df.copy()
  if bairros:
      df_filtered = df_filtered[df_filtered["bairro"].isin(bairros)]

  option_map = st.selectbox(
      "Tipo de crime",
      sorted(df["tipo_crime"].dropna().unique())
  )

  df_tipo_crime = df_filtered[df_filtered["tipo_crime"] == option_map].copy()
  
  df_expanded = df_tipo_crime.loc[df_tipo_crime.index.repeat(df_tipo_crime["quantidade_vitimas"])].reset_index(drop=True)

  fig = px.density_map(
      df_expanded,
      lat="latitude",
      lon="longitude",
      radius=15,
      center={"lat": -8.05, "lon": -34.9},
      zoom=11,
      title="Mapa de Calor"
  )

  fig.update_layout(height=700)
  st.title("Mapa de Calor")
  st.markdown("Este mapa mostra a concentração de **homicídios** por bairro em Recife, com peso baseado no número de vítimas registradas.")

  st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
