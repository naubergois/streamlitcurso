# -*- coding: utf-8 -*-
"""StreamlitExercicio1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1oZP1nIUU3eofMB-tMQB3Sno2Xqaso1rr
"""

# Exercícios com Streamlit para exibição de DataFrames

# Exercício 1: Exibir um DataFrame usando st.write
# -------------------------------------------------
# Descrição: Neste exercício, você deve criar um DataFrame simples
# e exibi-lo utilizando a função st.write do Streamlit.

import streamlit as st
import pandas as pd

# Criando o DataFrame
data = {
    'Nome': ['Alice', 'Bob', 'Charlie'],
    'Idade': [25, 30, 35],
    'Profissão': ['Engenheira', 'Médico', 'Advogado']
}
df = pd.DataFrame(data)

# Usando st.write para exibir o DataFrame
st.write("Exemplo de DataFrame utilizando `st.write`:")
st.write(df)

# Exercício 2: Exibir um DataFrame usando st.dataframe
# ----------------------------------------------------
# Descrição: Neste exercício, você deve criar outro DataFrame
# e exibi-lo utilizando a função st.dataframe do Streamlit.

# Criando um novo DataFrame
data2 = {
    'Produto': ['Notebook', 'Teclado', 'Mouse'],
    'Preço': [3000, 150, 80],
    'Quantidade': [5, 10, 15]
}
df2 = pd.DataFrame(data2)

# Usando st.dataframe para exibir o DataFrame
st.write("Exemplo de DataFrame utilizando `st.dataframe`:")
st.dataframe(df2)

# Exercício 3: Exibir uma tabela usando st.table
# ----------------------------------------------
# Descrição: Neste exercício, você deve criar um terceiro DataFrame
# e exibi-lo como uma tabela utilizando a função st.table do Streamlit.

# Criando um terceiro DataFrame
data3 = {
    'Cidade': ['São Paulo', 'Rio de Janeiro', 'Curitiba'],
    'População': [12300000, 6748000, 1949000],
    'Área (km²)': [1521, 1200, 430]
}
df3 = pd.DataFrame(data3)

# Usando st.table para exibir a tabela
st.write("Exemplo de DataFrame utilizando `st.table`:")
st.table(df3)