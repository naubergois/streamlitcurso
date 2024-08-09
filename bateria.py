import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import time
import random
import string

# Configurando estilo do Seaborn
sns.set(style="whitegrid")

# Função para gerar dados fictícios de baterias
def generate_fake_battery_data(num_samples=1000):

    data = {
        'Voltage': np.random.uniform(10, 15, num_samples),
        'Current': np.random.uniform(100, 500, num_samples),
        'Temperature': np.random.uniform(20, 40, num_samples),
        'Capacity': np.random.uniform(20, 100, num_samples),
        'Health': np.random.choice(['Good', 'Replace Soon', 'Replace Now'], num_samples, p=[0.7, 0.2, 0.1])
    }
    return pd.DataFrame(data)

# Função para gerar um novo carro
def generate_new_car():
    plates = ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))
    model = random.choice(['Sedan', 'SUV', 'Truck', 'Coupe', 'Convertible'])
    battery_type = random.choice(['Lithium-Ion', 'Nickel-Metal Hydride', 'Lead-Acid', 'Solid-State'])
    return {
        'Plate': plates,
        'Model': model,
        'Battery Type': battery_type
    }

# Função para treinar o modelo de Machine Learning
def train_model(data):
    X = data[['Voltage', 'Current', 'Temperature', 'Capacity']]
    y = data['Health']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test

# Função para fazer predições com novos dados
def predict_new_data(model, new_data):
    X_new = new_data[['Voltage', 'Current', 'Temperature', 'Capacity']]
    y_pred = model.predict(X_new)
    return y_pred

# Função para exibir os resultados na tela
def display_results(new_data, y_pred, car_info):
    st.write("## Informações do Novo Carro")
    st.write(f"**Placa:** {car_info['Plate']}")
    st.write(f"**Modelo:** {car_info['Model']}")
    st.write(f"**Tipo de Bateria:** {car_info['Battery Type']}")
    
    st.write("## Predição para a Bateria do Novo Carro")
    st.write(new_data)
    st.write(f"**Predição:** {y_pred[0]}")

    st.write("## Gráfico de Tensão por Saúde da Bateria")
    fig, ax = plt.subplots()
    sns.barplot(x=new_data.index, y=new_data['Voltage'], ax=ax, palette='viridis')
    ax.set_ylabel("Tensão (V)")
    ax.set_xlabel("Nova Bateria")
    st.pyplot(fig)

# Streamlit App
def main():
    st.title("Análise de Baterias de Carros com Predição Contínua")

    # Inicializando dados e modelo
    if 'data' not in st.session_state:
        st.session_state['data'] = generate_fake_battery_data()
    if 'model' not in st.session_state:
        st.session_state['model'], st.session_state['X_test'], st.session_state['y_test'] = train_model(st.session_state['data'])

    # Loop para simulação contínua de novos dados
    while True:
        # Gerar um novo carro
        car_info = generate_new_car()

        # Gerar novos dados de teste
        st.session_state['new_test_data'] = generate_fake_battery_data(num_samples=1)
        
        # Fazer predição com os novos dados de teste
        y_pred = predict_new_data(st.session_state['model'], st.session_state['new_test_data'])
        
        # Exibir os resultados no dashboard
        display_results(st.session_state['new_test_data'], y_pred, car_info)
        
        # Esperar 10 segundos antes de gerar o próximo carro
        time.sleep(10)
        
        # Atualizar o dashboard
        st.rerun()

if __name__ == '__main__':
    main()
