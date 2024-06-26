import pandas as pd
from datetime import datetime


def abre_e_normaliza_cliente(cliente_file):
    clientes = pd.read_csv(cliente_file, sep=';')
    # Converter colunas de data para datetime, ajustando o formato e tratando valores inválidos
    clientes['data_nascimento'] = pd.to_datetime(
        clientes['data_nascimento'],
        format='%Y-%m-%d',
        errors='coerce'
    )

    # Filtrar clientes com datas de nascimento inválidas
    clientes = clientes[clientes['data_nascimento'].notna()]

    # Tratar dados ausentes e inválidos
    clientes.fillna({'cidade': 'Desconhecida', 'setor': 'Desconhecido'}, inplace=True)

    # Adicionar coluna de idade no dataset de clientes
    clientes['idade'] = (datetime.now() - clientes['data_nascimento']).dt.days // 365

    return clientes


def abre_e_normaliza_vendas(vendas_file):
    vendas = pd.read_csv(vendas_file, sep=';', parse_dates=['DATA'])
    # Converter colunas de data para datetime, ajustando o formato e tratando valores inválidos
    vendas['DATA'] = pd.to_datetime(vendas['DATA'], errors='coerce')

    # Filtrar vendas com datas inválidas
    vendas = vendas[vendas['DATA'].notna()]

    # Normalizar os valores de vendas
    vendas['valor'] = vendas['valor'].replace(',', '.', regex=True).astype(float)

    return vendas


def abre_e_normaliza_produtos(produtos_file):
    produtos = pd.read_csv(produtos_file, sep=';')
    # Normalizar os valores de produtos
    produtos['preco_atual'] = produtos['preco_atual'].replace(',', '.', regex=True).astype(float)

    return produtos


def abre_e_normaliza_produtos_vendidos(produtos_vendidos_file):
    produtos_vendidos = pd.read_csv(produtos_vendidos_file, sep=';')
    # Converter colunas numéricas e tratar valores inválidos
    produtos_vendidos['quantidade'] = produtos_vendidos['quantidade'].astype(float)
    produtos_vendidos['valor_unitario'] = produtos_vendidos['valor_unitario'].astype(float)
    produtos_vendidos['desconto_aplicado'] = produtos_vendidos['desconto_aplicado'].astype(float)

    return produtos_vendidos


def abre_e_normaliza_forma_pagamento(forma_pagamento_file):
    formas_pagamento = pd.read_csv(forma_pagamento_file, sep=';')
    # Tratar dados de formas de pagamento
    formas_pagamento['avista'] = formas_pagamento['avista'].map({'S': True, 'N': False})
    formas_pagamento['parcelas'] = formas_pagamento['parcelas'].astype(int)
    return formas_pagamento


def prepara_vendas_mensais(vendas):
    # Agrupar vendas por mês
    vendas['DATA'] = pd.to_datetime(vendas['DATA'])
    vendas_mensais = vendas.resample('M', on='DATA').sum()['valor'].reset_index()
    vendas_mensais.columns = ['ds', 'real']

    # Adicionar características temporais
    vendas_mensais['month'] = vendas_mensais['ds'].dt.month
    vendas_mensais['year'] = vendas_mensais['ds'].dt.year
    vendas_mensais['dayofweek'] = vendas_mensais['ds'].dt.dayofweek
    return vendas_mensais


def prepara_vendas_por_produto(produtos, produtos_vendidos):
    vendas_por_produto =  produtos_vendidos.groupby('codigo_produto')['valor_unitario'].sum().reset_index()
    dados_completos = pd.merge(vendas_por_produto, produtos, left_on='codigo_produto', right_on='codigo_produto',
                               how='left')
    return dados_completos
