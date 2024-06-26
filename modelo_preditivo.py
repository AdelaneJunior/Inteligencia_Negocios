from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from fbprophet import Prophet


def modelo(vendas, produtos, produtos_vendidos, clientes):
    # Agregar dados de vendas por cidade e produto
    vendas_produtos = pd.merge(vendas, produtos_vendidos, on='numero_venda')
    vendas_produtos = pd.merge(vendas_produtos, produtos, on='codigo_produto')
    vendas_produtos = pd.merge(vendas_produtos, clientes, left_on='cliente', right_on="Codigo")

    # Agregar os dados de vendas por produto
    vendas_por_produto = vendas_produtos.groupby('produto').agg({'valor_unitario': 'sum',
                                                                 'custo_atual': 'first',
                                                                 'preco_atual': 'first',
                                                                 'secao': 'first'}).reset_index()

    # Preparar dados para o modelo
    X = vendas_por_produto[['produto', 'custo_atual', 'preco_atual', 'secao']]
    y = vendas_por_produto['valor_unitario']

    # Aplicar One-Hot Encoding para variáveis categóricas
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), ['produto', 'secao'])],
                           remainder='passthrough')
    X_encoded = ct.fit_transform(X)

    # Dividir dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Criar modelo de regressão linear
    modelo = LinearRegression()

    # Treinar o modelo
    modelo.fit(X_train, y_train)

    # Fazer previsões
    y_pred = modelo.predict(X_test)

    # Avaliar o modelo
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error (MSE): {mse}')
    print(f'R-squared (R²): {r2}')

    # Exemplo de previsão para novos dados
    # Suponha que temos novos dados para prever
    novo_produto = 'TOALHA C/CAPUZ'
    nova_secao = 'Seção B'  # Nova categoria que não estava presente no treinamento
    novos_dados = pd.DataFrame([[novo_produto, 10.0, 15.0, nova_secao]],
                               columns=['produto', 'custo_atual', 'preco_atual', 'secao'])

    # Transformar os novos dados com One-Hot Encoding
    novos_dados_encoded = ct.transform(novos_dados)

    previsao = modelo.predict(novos_dados_encoded)
    print(f'Previsão de vendas futuras para {novo_produto} na {nova_secao}: {previsao}')

    return modelo
    # for lag in range(1, 13):
    #     vendas_mensais[f'lag_{lag}'] = vendas_mensais['real'].shift(lag)
    #
    # vendas_mensais['rolling_mean_3'] = vendas_mensais['real'].rolling(window=3).mean().shift(1)
    # vendas_mensais['rolling_mean_6'] = vendas_mensais['real'].rolling(window=6).mean().shift(1)
    # vendas_mensais['rolling_mean_12'] = vendas_mensais['real'].rolling(window=12).mean().shift(1)
    #
    # vendas_mensais.dropna(inplace=True)
    #
    # X = vendas_mensais.drop(['ds', 'real'], axis=1)
    # y = vendas_mensais['real']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    #
    # param_grid = {
    #     'n_estimators': [100, 200],
    #     'max_depth': [10, 20],
    #     'min_samples_split': [2, 5],
    #     'min_samples_leaf': [1, 2]
    # }
    #
    # grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5,
    #                            scoring='neg_mean_squared_error')
    # grid_search.fit(X_train, y_train)
    #
    # best_model = grid_search.best_estimator_
    #
    # y_pred = best_model.predict(X_test)
    #
    # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # print(f'RMSE: {rmse}')
    #
    # future_dates = [vendas_mensais['ds'].max() + timedelta(days=30 * i) for i in range(1, 13)]
    #
    # future_df = pd.DataFrame(future_dates, columns=['ds'])
    # future_df['month'] = future_df['ds'].dt.month
    # future_df['year'] = future_df['ds'].dt.year
    # future_df['dayofweek'] = future_df['ds'].dt.dayofweek
    #
    # last_known_values = vendas_mensais.tail(12).set_index('ds')
    # for i in range(12):
    #     last_row = last_known_values.tail(1).copy()
    #     last_row['ds'] = future_df.iloc[i]['ds']
    #     last_row['month'] = future_df.iloc[i]['month']
    #     last_row['year'] = future_df.iloc[i]['year']
    #     last_row['dayofweek'] = future_df.iloc[i]['dayofweek']
    #
    #     for lag in range(1, 13):
    #         if i - lag < 0:
    #             last_row[f'lag_{lag}'] = vendas_mensais.iloc[i - lag]['real']
    #         else:
    #             last_row[f'lag_{lag}'] = future_df.iloc[i - lag]['real']
    #
    #     last_row['rolling_mean_3'] = last_known_values['real'].rolling(window=3).mean().iloc[-1]
    #     last_row['rolling_mean_6'] = last_known_values['real'].rolling(window=6).mean().iloc[-1]
    #     last_row['rolling_mean_12'] = last_known_values['real'].rolling(window=12).mean().iloc[-1]
    #
    #     last_row_X = last_row.drop(['ds', 'real'], axis=1)
    #     future_df.loc[i, 'real'] = best_model.predict(last_row_X)[0]
    #
    #     last_known_values = pd.concat([last_known_values, last_row.set_index('ds')])
    #
    # vendas_mensais['previsao'] = np.nan
    # future_df['previsao'] = future_df['real']
    #
    # fig = px.line(vendas_mensais, x='ds', y=['real', 'previsao'], labels={'value': 'Valor das Vendas', 'ds': 'Data'},
    #               title='Previsões de Vendas com Random Forest (Melhorado)',
    #               color_discrete_map={'real': 'blue', 'previsao': 'red'})
    #
    # fig.add_trace(px.line(future_df, x='ds', y='previsao', color_discrete_sequence=['green']).data[0])
    #
    # return fig
