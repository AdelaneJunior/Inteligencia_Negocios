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
    # Preparar os dados para a regressão linear
    X = vendas_por_ano['ANO'].values.reshape(-1, 1)
    y = vendas_por_ano['Vendas'].values

    # Criar e treinar o modelo de regressão linear
    modelo = LinearRegression()
    modelo.fit(X, y)

    previsoes_treino = modelo.predict(X)

    # Calcular o Mean Absolute Error
    mae = mean_absolute_error(y, previsoes_treino)

    print(f'Mean Absolute Error (MAE): {mae:.2f}')

    # Fazer previsões
    anos_futuros = np.array([2023,2024, 2025, 2026]).reshape(-1, 1)
    previsoes = modelo.predict(anos_futuros)

    # Adicionar as previsões ao DataFrame
    df_previsoes = pd.DataFrame({'ANO': anos_futuros.flatten(), 'Vendas': previsoes})

    # Plotar o gráfico com a linha de regressão e previsões
    fig = px.line(vendas_por_ano, x='ANO', y='Vendas', title='Quantidade de Vendas por Ano com Regressão Linear',
                  markers=True)
    fig.add_scatter(x=df_previsoes['ANO'], y=df_previsoes['Vendas'], mode='lines+markers', name='Previsões')
    fig.add_scatter(x=vendas_por_ano['ANO'], y=modelo.predict(X), mode='lines', name='Regressão Linear',
                    line=dict(dash='dash'))
    return fig

# def realiza_previsao(vendas, produtos_vendidos):
#     vendas_produtos = vendas.merge(produtos_vendidos, on='numero_venda')
#
#     # Agrupando as vendas por data para obter o valor total das vendas diárias
#     vendas_diarias = vendas_produtos.groupby('DATA').agg({'numero_venda': 'count'}).reset_index()
#
#     print(vendas_diarias.head())
#     # Renomeando as colunas para clareza
#     vendas_diarias.columns = ['data', 'quantidade_vendas']
#
#     # Adicionando features adicionais
#     vendas_diarias['data_ordinal'] = vendas_diarias['data'].map(pd.Timestamp.toordinal)
#     vendas_diarias['month'] = vendas_diarias['data'].dt.month
#     vendas_diarias['day_of_week'] = vendas_diarias['data'].dt.dayofweek
#
#     return treina_modelo(vendas_diarias)
#
#
# def treina_modelo(vendas_diarias):
#     # Separando variáveis independentes (X) e dependentes (y)
#     X = vendas_diarias[['data_ordinal', 'month', 'day_of_week']]
#     y = vendas_diarias['valor_vendas']
#
#     # Dividindo os dados em treino e teste (utilizando corte temporal)
#     split_date = pd.Timestamp('2020-12-31').toordinal()  # Ajuste essa data conforme necessário
#     X_train = X[X['data_ordinal'] <= split_date].copy()
#     X_test = X[X['data_ordinal'] > split_date].copy()
#     y_train = y[X['data_ordinal'] <= split_date].copy()
#     y_test = y[X['data_ordinal'] > split_date].copy()
#
#     # Padronizando os dados
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#
#     # Ajustando o modelo de regressão linear
#     modelo = LinearRegression()
#     modelo.fit(X_train_scaled, y_train)
#
#     # Fazendo previsões com o conjunto de teste
#     y_pred = modelo.predict(X_test_scaled)
#
#     # Avaliando o modelo
#     mae = mean_absolute_error(y_test, y_pred)
#     print(f'Mean Absolute Error: {mae}')
#
#     # Criando DataFrame para visualização das previsões
#     X_test.loc[:, 'valor_vendas_real'] = y_test
#     X_test.loc[:, 'valor_vendas_pred'] = y_pred  # Usando .loc para evitar o aviso
#
#     # Convertendo data_ordinal de volta para datetime para plotagem
#     X_test['data'] = X_test['data_ordinal'].map(pd.Timestamp.fromordinal)
#
#     # Plotando as previsões vs. valores reais
#     fig = px.scatter(X_test, x='data', y='valor_vendas_real', title='Previsões vs. Valores Reais',
#                      labels={'data': 'Data', 'valor_vendas_real': 'Valor das Vendas'})
#     fig.add_scatter(x=X_test['data'], y=X_test['valor_vendas_pred'], mode='lines', name='Previsões')
#
#     last_date = vendas_diarias['data'].max()
#     future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 91)]
#     future_dates_ordinal = [date.toordinal() for date in future_dates]
#
#     # Criando dataframe para datas futuras
#     future_dates_df = pd.DataFrame({
#         'data_ordinal': future_dates_ordinal,
#         'month': [date.month for date in future_dates],
#         'day_of_week': [date.dayofweek for date in future_dates]
#     })
#
#     # Padronizando as datas futuras
#     future_dates_scaled = scaler.transform(future_dates_df)
#
#     # Fazendo previsões para os próximos 3 meses
#     future_forecast = modelo.predict(future_dates_scaled)
#
#     # Criando dataframe para previsões futuras
#     future_df = pd.DataFrame({'data': future_dates, 'valor_vendas': future_forecast})
#
#     #
#     fig = px.line(future_df, x='data', y='valor_vendas', title='Previsão das Vendas para os Próximos 3 Meses',
#                   labels={'data': 'Data', 'valor_vendas': 'Valor das Vendas'})
#
#     fig = px.line(vendas_diarias, x='data', y='valor_vendas',
#                   title='Previsão das Vendas para os Próximos 3 Meses com Regressão Linear',
#                   labels={'data': 'Data', 'valor_vendas': 'Valor das Vendas'})
#
#     fig.add_trace(go.Scatter(x=future_df['data'], y=future_df['valor_vendas'],
#                              mode='lines', name='Previsão Futura'))
#     return fig
