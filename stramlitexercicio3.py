import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np

# Configuração da página
st.set_page_config(layout="wide")
st.title("Mapa de Calor dos Casos de Dengue no Ceará")

# Carregar o shapefile dos municípios do Ceará
@st.cache_data
def load_geodata():
    # Substitua 'CE_UF_2022/ceara_municipios.shp' pelo caminho para o seu shapefile dos municípios do Ceará
    geodata = gpd.read_file('CE_UF_2022/CE_UF_2022.shp')
    return geodata

geodata = load_geodata()

# Gerar dados fictícios de casos de dengue por município
np.random.seed(42)  # Para reprodutibilidade
geodata['Casos_Dengue'] = np.random.randint(50, 500, size=len(geodata))

# Filtros interativos
st.sidebar.header("Filtros")

# Filtro por município
selected_municipios = st.sidebar.multiselect(
    "Selecione os municípios para exibir:",
    options=geodata['NM_MUN'],
    default=geodata['NM_MUN']
)

# Filtro por número de casos
min_cases, max_cases = st.sidebar.slider(
    "Selecione o intervalo de casos de dengue:",
    int(geodata['Casos_Dengue'].min()), int(geodata['Casos_Dengue'].max()), 
    (int(geodata['Casos_Dengue'].min()), int(geodata['Casos_Dengue'].max()))
)

# Aplicar filtros ao GeoDataFrame
filtered_geodata = geodata[(geodata['NM_MUN'].isin(selected_municipios)) & 
                           (geodata['Casos_Dengue'] >= min_cases) & 
                           (geodata['Casos_Dengue'] <= max_cases)]

# Exibir tabela dos casos de dengue filtrados
st.subheader("Tabela de Casos Fictícios de Dengue por Município (Filtrados)")
st.dataframe(filtered_geodata[['NM_MUN', 'Casos_Dengue']])

# Criar mapa de calor
st.subheader("Mapa de Calor dos Casos de Dengue (Filtrados)")

# Centro do Ceará para centralizar o mapa
centro_ceara = [-5.4984, -39.3206]
m = folium.Map(location=centro_ceara, zoom_start=7)

# Adicionar os polígonos dos municípios ao mapa
for _, row in filtered_geodata.iterrows():
    sim_geo = gpd.GeoSeries(row['geometry']).simplify(tolerance=0.001)
    geo_json = sim_geo.to_json()
    folium.GeoJson(data=geo_json, style_function=lambda x: {
        'fillColor': 'yellow',
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.3,
    }).add_to(m)

# Preparar dados para o heatmap
heat_data = []

for _, row in filtered_geodata.iterrows():
    # Obter o centroide do município
    centroid = row['geometry'].centroid
    heat_data.append([centroid.y, centroid.x, row['Casos_Dengue']])

# Adicionar camada de heatmap
from folium.plugins import HeatMap

HeatMap(heat_data, radius=25, blur=15, max_zoom=10).add_to(m)

# Exibir o mapa no Streamlit
st_data = st_folium(m, width=700, height=500)
