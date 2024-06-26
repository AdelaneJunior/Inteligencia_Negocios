import plotly.express as px
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import datetime as dt


def realiza_previsao(vendas, produtos_vendidos):
    vendas['ANO'] = vendas['DATA'].dt.year

    vendas_por_ano = vendas.groupby('ANO').size().reset_index(name='Vendas')

    fig = px.line(vendas_por_ano, x='ANO', y='Vendas', title='Quantidade de Vendas por Ano', markers=True)

    print(vendas['DATA'].max())
    return treina_modelo(vendas_por_ano)


def treina_modelo(vendas_por_ano):

    X = vendas_por_ano['ANO'].values.reshape(-1, 1)
    y = vendas_por_ano['Vendas'].values

    modelo = LinearRegression()
    modelo.fit(X, y)

    previsoes_treino = modelo.predict(X)

    mae = mean_absolute_error(y, previsoes_treino)

    print(f'Mean Absolute Error (MAE): {mae:.2f}')

    anos_futuros = np.array([2023,2024, 2025, 2026]).reshape(-1, 1)
    previsoes = modelo.predict(anos_futuros)

    df_previsoes = pd.DataFrame({'ANO': anos_futuros.flatten(), 'Vendas': previsoes})

    fig = px.line(vendas_por_ano, x='ANO', y='Vendas', title='Quantidade de Vendas por Ano com Regressão Linear',
                  markers=True)
    fig.add_scatter(x=df_previsoes['ANO'], y=df_previsoes['Vendas'], mode='lines+markers', name='Previsões')
    fig.add_scatter(x=vendas_por_ano['ANO'], y=modelo.predict(X), mode='lines', name='Regressão Linear',
                    line=dict(dash='dash'))
    return fig
