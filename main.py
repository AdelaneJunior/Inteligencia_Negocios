import plotly.express as px
from dash import Dash, dcc, html
import processamento_dados as prcd
import previsao as prev


clientes = prcd.abre_e_normaliza_cliente('csv/clientes.csv')
vendas = prcd.abre_e_normaliza_vendas('csv/venda.csv')
produtos = prcd.abre_e_normaliza_produtos('csv/produtos.csv')
produtos_vendidos = prcd.abre_e_normaliza_produtos_vendidos('csv/produtos_vendidos.csv')
formas_pagamento = prcd.abre_e_normaliza_forma_pagamento('csv/formas_pagamento.csv')
vendas_por_produto = prcd.prepara_vendas_por_produto(produtos, produtos_vendidos)


def vendas_timeline(vendas):
    vendas_por_data = vendas.groupby(vendas['DATA'].dt.to_period('M')).agg({'valor': 'sum'}).reset_index()
    vendas_por_data['DATA'] = vendas_por_data['DATA'].dt.to_timestamp()
    fig = px.line(vendas_por_data, x='DATA', y='valor', title='Vendas Mensais ao Longo do Tempo')
    return fig


def scatter_preco_custo(produtos):
    fig = px.scatter(produtos, x='custo_atual', y='preco_atual', color='secao', title='Preço vs Custo dos Produtos')
    return fig


def vendas_por_cidade(vendas, clientes):
    vendas_com_clientes = vendas.merge(clientes, left_on='cliente', right_on='Codigo', how='left')
    vendas_por_cidade = vendas_com_clientes.groupby('cidade').agg({'Codigo': 'count'}).reset_index()
    fig = px.bar(vendas_por_cidade, x='cidade', y='Codigo', title='Vendas por Cidade')
    return fig


def secao_mais_vendida(produtos, produtos_vendidos):
    produtos_com_produtos_vendidos = produtos.merge(produtos_vendidos, on='codigo_produto')
    produtos_mais_vendidos = produtos_com_produtos_vendidos.groupby('secao').agg({'codigo_produto':'count'}).reset_index()

    produtos_com_produtos_vendidos = produtos.merge(produtos_vendidos, on='codigo_produto')

    produtos_mais_vendidos = produtos_com_produtos_vendidos.groupby('secao').agg(
        {'codigo_produto': 'count'}).reset_index()
    produtos_mais_vendidos.rename(columns={'codigo_produto': 'quantidade_vendida'}, inplace=True)

    produtos_mais_vendidos = produtos_mais_vendidos.sort_values(by='quantidade_vendida', ascending=True)

    fig = px.bar(produtos_mais_vendidos, x='quantidade_vendida', y='secao', title='Sessões Mais Vendidas',
                 labels={'secao': 'Seção', 'quantidade_vendida': 'Quantidade Vendida'})

    return fig


app = Dash(__name__)

app.layout = html.Div([
    html.H1('Painel de Análise de Vendas'),
    dcc.Graph(id='vendas-timeline', figure=vendas_timeline(vendas)),
    dcc.Graph(id='scatter-preco-custo', figure=scatter_preco_custo(produtos)),
    dcc.Graph(id='vendas-por-cidade', figure=vendas_por_cidade(vendas, clientes)),
    dcc.Graph(id='secao-mais-vendida', figure=secao_mais_vendida(produtos, produtos_vendidos), style={'width': '100%',
                                                                                                      'height': '600px'}
              ),
    dcc.Graph(id='previsao', figure=prev.realiza_previsao(vendas, produtos_vendidos))
])

if __name__ == '__main__':
    app.run_server(debug=True)

