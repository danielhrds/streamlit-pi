import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Dashboard de Gráficos", layout="wide")

try:
    df_train = pd.read_csv("dataset_ocorrencias_delegacia_5.csv")
except FileNotFoundError:
    df_train = None

st.sidebar.title("Upload de Dataset")
uploaded_file = st.sidebar.file_uploader("Carregue um CSV para teste", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset de teste carregado!")
elif df_train is not None:
    df = df_train.copy()
    st.info("Mostrando dados do dataset de treino. Carregue um arquivo para analisar novos dados.")
else:
    st.error("Dataset de treino não encontrado. Por favor, adicione 'dataset_ocorrencias_delegacia_5.csv' à pasta ou carregue um novo arquivo.")
    st.stop()

df_roubos = df[df["tipo_crime"].str.lower() == "roubo"].copy()
df_roubos["data_ocorrencia"] = pd.to_datetime(df_roubos["data_ocorrencia"])
df_roubos["ano_mes"] = df_roubos["data_ocorrencia"].dt.to_period("M").astype(str)
df_roubos['hora'] = df_roubos['data_ocorrencia'].dt.hour

dias_ordenados = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_roubos['dia_semana'] = df_roubos['data_ocorrencia'].dt.day_name()
df_roubos['dia_semana'] = pd.Categorical(df_roubos['dia_semana'], categories=dias_ordenados, ordered=True)

total_ocorrencias = len(df_roubos)
n_bairros = df_roubos["bairro"].nunique()

# anomalias
df_ano = df_roubos.groupby("ano_mes")["quantidade_vitimas"].sum().reset_index()
scaler = StandardScaler()
X = scaler.fit_transform(df_ano[["quantidade_vitimas"]])
iso = IsolationForest(contamination=0.1, random_state=42)
df_ano["anomaly"] = iso.fit_predict(X)
n_anomalias = (df_ano["anomaly"] == -1).sum()

st.markdown("## Dashboard de Roubos em Recife")
col1, col2, col3 = st.columns(3)
col1.metric("Total de Ocorrências", f"{total_ocorrencias:,}")
col2.metric("Anomalias", f"{n_anomalias}")
col3.metric("Bairros Afetados", f"{n_bairros}")
st.markdown("---")


st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    h1, h2, h3 { color: #1f77b4; }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title(" Menu de Navegação")

option = st.sidebar.radio(
    "Selecione a página:",
    (
        "Gráficos relacionados a roubo",
        "Mapa de calor",
        "Clusters",
        "Anomalias"
    ),
    index=0
)

if option == "Gráficos relacionados a roubo":
    st.header("Análises de Roubos em Recife")

    fig1 = px.line(
        df_ano,
        x="ano_mes",
        y="quantidade_vitimas",
        title="Evolução de Roubos ao Longo do Tempo",
        labels={"quantidade_vitimas": "Nº de Vítimas", "ano_mes": "Ano-Mês"},
    )
    fig1.update_traces(line=dict(width=3, color="#1f77b4"))
    st.plotly_chart(fig1, use_container_width=True)

    # roubos por bairro
    ranking_bairro = (
        df_roubos["bairro"].value_counts().reset_index()
        .rename(columns={"index": "bairro", "bairro": "ocorrencias"})  # <- corrigido
        .head(15)
    )
    fig2 = px.bar(
        ranking_bairro,
        y="ocorrencias",
        x="count",
        orientation="h",
        title="Top 10 Bairros com Mais Ocorrências de Roubo",
        labels={"ocorrencias": "Quantidade de Ocorrências", "bairro": "Bairro"},
        color="ocorrencias",
        color_continuous_scale="Viridis"
    )
    fig2.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig2, use_container_width=True)

    # roubos por hora do dia
    if "hora" in df_roubos.columns:
        fig3 = px.histogram(
            df_roubos,
            x="hora",
            title="Distribuição de Roubos por Hora do Dia",
            labels={"hora": "Hora do Dia", "count": "Quantidade de Ocorrências"},
            color_discrete_sequence=["#9467bd"]
        )
        st.plotly_chart(fig3, use_container_width=True)

    # roubos por dia da semana
    if "dia_semana" in df_roubos.columns:
        fig4 = px.histogram(
            df_roubos.sort_values("dia_semana"),
            x="dia_semana",
            title="Quantidade de Roubos por Dia da Semana",
            labels={"dia_semana": "Dia da Semana", "count": "Quantidade de Ocorrências"},
            color_discrete_sequence=["#17becf"]
        )
        st.plotly_chart(fig4, use_container_width=True)

    # tipos de armas mais usadas
    if "arma_utilizada" in df_roubos.columns:
        ranking_arma = (
            df_roubos["arma_utilizada"].value_counts().reset_index()
            .rename(columns={"index": "arma_utilizada", "arma_utilizada": "ocorrencias"})
            .head(10)
        )
        fig5 = px.bar(
            ranking_arma,
            y="ocorrencias",
            x="count",
            orientation="h",
            title="Tipos de Armas Mais Utilizadas em Roubos",
            labels={"ocorrencias": "Quantidade de Ocorrências", "arma_utilizada": "Tipo de Arma"},
            color="ocorrencias",
            color_continuous_scale="Magma"
        )
        fig5.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig5, use_container_width=True)

    ranking = df_roubos.groupby("bairro")["quantidade_vitimas"].sum().reset_index()
    ranking = ranking.sort_values(by="quantidade_vitimas", ascending=False).head(10)

    fig = px.bar(
        ranking, 
        x="bairro", 
        y="quantidade_vitimas",
        text="quantidade_vitimas",
        color_discrete_sequence=["#1f77b4"],
        title=" Top 10 Bairros com Mais Vítimas de Roubo"
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
elif option == "Mapa de calor":
    st.subheader("Mapa de Calor")

    # Adiciona a opção "Todos" à lista de meses
    selectable_months = ["Todos"] + sorted(df_roubos["ano_mes"].unique())
    selected_month = st.selectbox("Selecionar mês:", options=selectable_months, index=0)

    bairros = st.multiselect(
        "Filtrar bairros:",
        options=sorted(df_roubos["bairro"].dropna().unique()),
        default=[]
    )

    selected_days = []
    df_filtered = df_roubos.copy()
    if bairros:
        df_filtered = df_filtered[df_filtered["bairro"].isin(bairros)]

    df_filtered = df_filtered[
        (df_filtered["tipo_crime"].str.lower() == "roubo")
    ].copy()

    # Aplica o filtro de mês apenas se "Todos" não estiver selecionado
    if selected_month != "Todos":
        df_filtered = df_filtered[df_filtered["ano_mes"] == selected_month]

    # filtros de hora e dia da semana
    if "hora" in df_roubos.columns:
        hour_min, hour_max = 0, 23
        selected_hour = st.slider("Filtrar por hora do dia:", min_value=hour_min, max_value=hour_max, value=(hour_min, hour_max))
        df_filtered = df_filtered[(df_filtered["hora"] >= selected_hour[0]) & (df_filtered["hora"] <= selected_hour[1])]
    if "dia_semana" in df_filtered.columns:
        selectable_days = sorted(df_roubos["dia_semana"].dropna().unique())
        selected_days = st.multiselect("Filtrar por dia da semana:", options=selectable_days, default=selectable_days)
        df_filtered = df_filtered[df_filtered["dia_semana"].isin(selected_days)]
        if selectable_days and not selected_days:
          df_filtered = df_roubos.copy()
          if bairros:
              df_filtered = df_filtered[df_filtered["bairro"].isin(bairros)]

          df_filtered = df_filtered[
              (df_filtered["tipo_crime"].str.lower() == "roubo")
          ].copy()
          df_filtered = df_filtered[df_filtered["dia_semana"].isin(selectable_days)]

    if not df_filtered.empty:
        df_expanded = df_filtered.loc[df_filtered.index.repeat(df_filtered["quantidade_vitimas"])].reset_index(drop=True)
        fig = px.density_map(
            df_expanded,
            lat="latitude",
            lon="longitude",
            radius=15,
            center={"lat": -8.05, "lon": -34.9},
            zoom=11,
            title=f"Mapa de Calor ({selected_month})",
        )
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"Este mapa mostra a concentração de **roubos** por bairro em Recife no mês de **{selected_month}**, com intensidade baseada no número de vítimas registradas.")
    else:
        st.warning("Nenhum dado de roubo encontrado para os filtros selecionados.")
elif option == "Clusters":
    st.header(" Agrupamento de Bairros por Perfis de Roubos")

    df_cluster = df_roubos.groupby("bairro").agg({
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
