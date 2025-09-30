import json
import os
from datetime import datetime
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import requests
import seaborn as sns
from matplotlib import ticker
from sklearn.linear_model import LinearRegression

"""
Autores:
    Alisson Santos da Silveira
    Felipe Pereira de Castro
    Gabriel Sperb Stoffel
    Karina Murta Starling Hayashi
"""

ANO_ATUAL = datetime.now().year
QTD_ANOS = 20
ULTIMOS_ANOS = [i for i in range(ANO_ATUAL - QTD_ANOS, ANO_ATUAL + 1)]
SIGLAS_ESTADOS = {
    'Acre': 'AC',
    'Alagoas': 'AL',
    'Amapá': 'AP',
    'Amazonas': 'AM',
    'Bahia': 'BA',
    'Ceará': 'CE',
    'Distrito Federal': 'DF',
    'Espírito Santo': 'ES',
    'Goiás': 'GO',
    'Maranhão': 'MA',
    'Mato Grosso': 'MT',
    'Mato Grosso do Sul': 'MS',
    'Minas Gerais': 'MG',
    'Pará': 'PA',
    'Paraíba': 'PB',
    'Paraná': 'PR',
    'Pernambuco': 'PE',
    'Piauí': 'PI',
    'Rio de Janeiro': 'RJ',
    'Rio Grande do Norte': 'RN',
    'Rio Grande do Sul': 'RS',
    'Rondônia': 'RO',
    'Roraima': 'RR',
    'Santa Catarina': 'SC',
    'São Paulo': 'SP',
    'Sergipe': 'SE',
    'Tocantins': 'TO'
}

db_user = os.getenv('DB_USER', 'user')
db_password = os.getenv('DB_PASSWORD', 'password')
db_name = os.getenv('DB_NAME', 'meu_banco')
db_port = os.getenv('DB_PORT', '5432')
db_host = os.getenv('DB_HOST', '5432')


def main():
    # PIB
    data_pib = obter_dados_pib()
    df_pib = transformar_dados_pib(data_pib)
    carregar_dados_pib(df_pib)
    plotar_graficos_pib(df_pib)

    # População
    path = obtem_dados_populacao()
    df_pop = transformar_dados_populacao(path)
    carrega_dados_populacao(df_pop)
    plotar_graficos_populacao(df_pop)

    # PIB per capta
    tabela_pib_per_capta = calcular_pib_per_capta(df_pib, df_pop)
    carrega_dados_pib_per_capta(tabela_pib_per_capta)
    plotar_graficos_pib_per_capta(tabela_pib_per_capta)


def plotar_graficos_pib_per_capta(tabela_pib_per_capta):
    """
    Gera gráficos de barras e previsões de regressão linear para o PIB per capita por estado e ano.

    Parâmetros:
        tabela_pib_per_capta (pd.DataFrame): DataFrame contendo as colunas 'SIGLA', 'ANO' e 'PIB_PER_CAPTA'.

    Funcionalidade:
        - Ordena os dados por PIB per capita e sigla do estado.
        - Gera um gráfico de barras comparativo do PIB per capita por estado e ano.
        - Para cada estado, realiza uma regressão linear para prever o PIB per capita até 2025.
        - Salva os gráficos gerados no diretório `output` com nomes específicos para cada estado.

    Dependências:
        - Bibliotecas: seaborn, matplotlib, sklearn.linear_model.LinearRegression, numpy.
        - O diretório `output` deve existir para salvar os gráficos.

    Exceções:
        - Certifique-se de que o DataFrame contém as colunas necessárias ('SIGLA', 'ANO', 'PIB_PER_CAPTA').
        - O diretório `output` deve estar acessível para salvar os arquivos.

    Saída:
        - Gráficos salvos no formato PNG no diretório `output`.
    """
    # Ordernação por valor e estado
    tabela_pib_per_capta = tabela_pib_per_capta.sort_values(by=['PIB_PER_CAPTA', 'SIGLA'], ascending=[False, True])
    # Configurando estilo
    sns.set_style(style="whitegrid")
    # Configurando dimenções
    plt.figure(figsize=(28, 10))
    # Criando as barras
    sns.barplot(x='SIGLA', y='PIB_PER_CAPTA', hue='ANO', data=tabela_pib_per_capta)
    # Rótulos e título
    plt.xlabel('Estados')
    plt.ylabel('PIB per capta')
    plt.title('Comparação PIB per capta')
    # Plotando
    plt.tight_layout()
    plt.savefig(f'output/pib_per_capta/comparação_pib_per_capta.png')
    plt.close()

    for sigla in tabela_pib_per_capta['SIGLA']:
        tabela_pib_per_capta_estado = tabela_pib_per_capta[tabela_pib_per_capta['SIGLA'] == sigla]
        # Regressão Linear
        lr = LinearRegression()

        # Reshape para o modelo
        X = tabela_pib_per_capta_estado['ANO'].values.reshape(-1, 1)
        # Valores de PIB per capta
        y = tabela_pib_per_capta_estado['PIB_PER_CAPTA'].values

        # Ajustando o modelo
        lr.fit(X, y)
        # Previsão
        y_predicts = lr.predict(X)

        # Previsão para os próximos anos
        anos_futuros = np.array([2022, 2023, 2024, 2025]).reshape(-1, 1)
        previsoes = lr.predict(anos_futuros)

        # Gráfico de PIB per capta
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue', label='Dados reais')
        plt.plot(X, y_predicts, color='orange', label='Regressão Linear')
        plt.plot(anos_futuros, previsoes, color='green', linestyle='--', marker='o', label='Previsão até 2025')
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.xlabel('Ano')
        plt.ylabel('PIB per capita')
        plt.title(f'PIB per capita - {sigla} (2022–2025)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'output/previsoes_pib_per_capta_estados/previsao_pib_per_capta_{sigla}.png')
        plt.close()


def calcular_pib_per_capta(df_pib, df_pop):
    """
    Calcula o PIB per capita com base nos dados de PIB e população.

    Parâmetros:
        df_pib (pd.DataFrame): DataFrame contendo os dados de PIB, com colunas 'SIGLA', 'ANO' e 'VALOR'.
        df_pop (pd.DataFrame): DataFrame contendo os dados de população, com colunas 'SIGLA', 'ANO' e 'VALOR'.

    Retorno:
        pd.DataFrame: DataFrame resultante contendo as colunas 'SIGLA', 'ANO', 'VALOR_PIB', 'VALOR_POP' e 'PIB_PER_CAPTA'.

    Funcionalidade:
        - Realiza o merge dos DataFrames de PIB e população com base nas colunas 'SIGLA' e 'ANO'.
        - Calcula o PIB per capita dividindo o valor do PIB pela população.
    """
    # Merge dos dados de PIB e população
    tabela_pib_per_capta = pd.merge(df_pib, df_pop, on=['SIGLA', 'ANO'], how='inner', suffixes=('_PIB', '_POP'))
    # Divindo o PIB pela população
    tabela_pib_per_capta['PIB_PER_CAPTA'] = tabela_pib_per_capta['VALOR_PIB'] / tabela_pib_per_capta['VALOR_POP']
    return tabela_pib_per_capta


def plotar_graficos_populacao(df_pop):
    """
    Gera gráficos comparativos da população por estado e ano.

    Parâmetros:
        df_pop (pd.DataFrame): DataFrame contendo as colunas 'SIGLA', 'ANO' e 'VALOR',
                               representando a sigla do estado, o ano e o número de pessoas.

    Funcionalidade:
        - Ordena os dados por número de pessoas e sigla do estado.
        - Gera um gráfico de barras comparativo da população entre estados para todos os anos.
        - Para cada ano, gera gráficos de barras individuais comparando a população entre estados.
        - Salva os gráficos gerados no diretório `output` com nomes específicos.

    Dependências:
        - Bibliotecas: seaborn, matplotlib.
        - O diretório `output` deve existir para salvar os gráficos.

    Exceções:
        - Certifique-se de que o DataFrame contém as colunas necessárias ('SIGLA', 'ANO', 'VALOR').
        - O diretório `output` deve estar acessível para salvar os arquivos.

    Saída:
        - Gráficos salvos no formato PNG no diretório `output`.
    """
    # Ordernação por valor e estado
    df_pop = df_pop.sort_values(by=['VALOR', 'SIGLA'], ascending=[False, True])
    # Configurando estilo
    sns.set_style(style="whitegrid")
    # Configurando dimenções
    plt.figure(figsize=(24, 10))
    # Criando as barras
    sns.barplot(x='SIGLA', y='VALOR', hue='ANO', data=df_pop)
    # Rótulos e título
    plt.xlabel('Estado')
    plt.ylabel('Número de Pessoas')
    plt.title('Comparação da população entre estados')
    # Plotando
    plt.tight_layout()
    plt.savefig(f'output/populacao_estados/comparacao_pop_estados.png')
    plt.close()

    for ano in ULTIMOS_ANOS:
        df_pop_ano = df_pop[df_pop['ANO'] == ano]

        # Ordernação por valor e estado
        df_pop_ano = df_pop_ano.sort_values(by=['VALOR', 'SIGLA'], ascending=[False, True])

        # Configurando estilo
        sns.set_style(style="whitegrid")

        # Configurando dimenções
        plt.figure(figsize=(16, 10))

        # Criando as barras
        sns.barplot(x='SIGLA', y='VALOR', data=df_pop_ano)

        # Rótulos e título
        plt.xlabel('Estado')
        plt.ylabel('Número de Pessoas')
        plt.title('Comparação da população entre estados (2021)')

        # Plotando
        plt.tight_layout()
        plt.savefig(f'output/populacao_estados_ano/comparacao_pop_estados_{ano}.png')
        plt.close()


def transformar_dados_populacao(path):
    """
    Transforma os dados de população a partir de um arquivo Excel.

    Parâmetros:
        path (str): Caminho para o arquivo Excel contendo os dados de população.

    Funcionalidade:
        - Lê o arquivo Excel e filtra os dados para incluir apenas registros com o sexo 'Ambos' e siglas válidas.
        - Remove colunas desnecessárias.
        - Agrega os dados por sigla dos estados.
        - Transforma os anos em uma única coluna chamada 'ANO' e os valores correspondentes em 'VALOR'.
        - Converte os anos para o tipo inteiro.
        - Filtra os dados para incluir apenas anos até o ano atual.

    Retorno:
        pd.DataFrame: DataFrame contendo as colunas 'SIGLA', 'ANO' e 'VALOR', representando a sigla do estado, o ano e o número de pessoas.

    Dependências:
        - Bibliotecas: pandas, openpyxl.
        - O arquivo Excel deve estar no formato esperado, com as colunas e estrutura adequadas.

    Exceções:
        - Certifique-se de que o arquivo fornecido existe e está acessível.
        - O arquivo deve conter as colunas necessárias para o processamento.
    """
    # Lendo o arquivo Excel
    df_pop = pd.read_excel(path, header=1, skiprows=4, engine='openpyxl')
    # Renomeando colunas
    df_pop = df_pop[(df_pop['SEXO'] == 'Ambos') & (~df_pop['SIGLA'].isin(['CO', 'ND', 'NO', 'SD', 'SU', 'BR']))]
    # Removendo colunas desnecessárias
    df_pop = df_pop.drop(columns=['IDADE', 'CÓD.', 'SEXO', 'LOCAL'])
    # Agregando por sigla
    df_pop = df_pop.groupby('SIGLA', as_index=False).sum()
    # Transpondo Anos para coluna ano
    df_pop = df_pop.melt(id_vars=['SIGLA'], var_name='ANO', value_name='VALOR')
    # Ano para inteiro
    df_pop['ANO'] = df_pop['ANO'].astype(int)
    # Anos até o ano atual
    df_pop = df_pop[df_pop['ANO'] <= ANO_ATUAL]
    return df_pop


def obtem_dados_populacao():
    """
    Faz o download de um arquivo Excel contendo dados de população a partir de um link do Google Drive.

    Funcionalidade:
        - Realiza uma requisição HTTP para baixar o arquivo Excel.
        - Salva o arquivo no diretório `data` com o nome `populacao.xlsx`.

    Retorno:
        str: Caminho para o arquivo salvo (`data/populacao.xlsx`).

    Dependências:
        - Biblioteca `requests` para realizar a requisição HTTP.
        - O diretório `data` deve existir para salvar o arquivo.

    Exceções:
        - Certifique-se de que o link do Google Drive é válido e acessível.
        - O diretório `data` deve estar acessível para salvar o arquivo.
    """
    file_id = '1xc40YIHHr_d9kQWjZ9i5eLE_OVzI701c'
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    response = requests.get(url)
    path = 'data/populacao.xlsx'
    with open(path, 'wb') as f:
        f.write(response.content)
    return path


def plotar_graficos_pib(df_pib):
    """
    Gera gráficos comparativos do PIB por estado e ano.

    Parâmetros:
        df_pib (pd.DataFrame): DataFrame contendo as colunas 'SIGLA', 'ANO' e 'VALOR',
                               representando a sigla do estado, o ano e o valor do PIB.

    Funcionalidade:
        - Ordena os dados por valor do PIB e sigla do estado.
        - Gera um gráfico de barras comparativo do PIB entre estados para todos os anos.
        - Para cada ano, gera gráficos de barras individuais comparando o PIB entre estados.
        - Salva os gráficos gerados no diretório `output` com nomes específicos.

    Dependências:
        - Bibliotecas: seaborn, matplotlib.
        - O diretório `output` deve existir para salvar os gráficos.

    Exceções:
        - Certifique-se de que o DataFrame contém as colunas necessárias ('SIGLA', 'ANO', 'VALOR').
        - O diretório `output` deve estar acessível para salvar os arquivos.

    Saída:
        - Gráficos salvos no formato PNG no diretório `output`.
    """
    # Ordernação por valor e estado
    df_pib = df_pib.sort_values(by=['VALOR', 'SIGLA'], ascending=[False, True])
    # Configurando estilo
    sns.set_style(style="whitegrid")
    # Configurando dimenções
    plt.figure(figsize=(24, 10))
    # Criando as barras
    sns.barplot(x='SIGLA', y='VALOR', hue='ANO', data=df_pib)
    # Rótulos e título
    plt.xlabel('Estado')
    plt.ylabel('Valor (em bilhões)')
    plt.title('Comparação PIB entre estados')
    # Plotando
    plt.tight_layout()
    plt.savefig('output/estados_pib/comparacao_pib_estados.png')
    plt.close()
    #####################
    for ano in ULTIMOS_ANOS:
        df_pib_ano = df_pib[df_pib['ANO'] == ano]

        # Ordernação por valor e estado
        df_pib_ano = df_pib_ano.sort_values(by=['VALOR', 'SIGLA'], ascending=[False, True])

        # Configurando estilo
        sns.set_style(style="whitegrid")

        # Configurando dimenções
        plt.figure(figsize=(16, 10))

        # Criando as barras
        sns.barplot(x='SIGLA', y='VALOR', data=df_pib_ano)

        # Rótulos e título
        plt.xlabel('Estado')
        plt.ylabel('Valor (em bilhões)')
        plt.title(f'Comparação PIB entre estados ({ano})')

        # Plotando
        plt.tight_layout()
        plt.savefig(f'output/estados_pib_ano/comparacao_pib_estados_{ano}.png')
        plt.close()


def transformar_dados_pib(data_pib):
    """
    Transforma os dados de PIB obtidos da API do IBGE em um DataFrame estruturado.

    Parâmetros:
        data_pib (list): Lista de dicionários contendo os dados de PIB no formato retornado pela API do IBGE.

    Funcionalidade:
        - Converte os dados JSON em um DataFrame do pandas.
        - Extrai o nome do estado a partir da chave 'localidade'.
        - Para cada ano nos últimos anos definidos, calcula o valor do PIB multiplicado por 100.
        - Remove colunas desnecessárias e mapeia os nomes dos estados para suas respectivas siglas.
        - Transforma os anos em uma única coluna chamada 'ANO' e os valores correspondentes em 'VALOR'.
        - Converte os anos para o tipo inteiro.

    Retorno:
        pd.DataFrame: DataFrame contendo as colunas 'SIGLA', 'ANO' e 'VALOR', representando a sigla do estado, o ano e o valor do PIB.

    Dependências:
        - Bibliotecas: pandas, json.
        - O dicionário `SIGLAS_ESTADOS` deve estar definido no escopo global.

    Exceções:
        - Certifique-se de que os dados fornecidos estão no formato esperado.
        - O DataFrame resultante deve conter as colunas necessárias para o processamento subsequente.
    """
    df_pib = pd.read_json(StringIO(json.dumps(data_pib)), orient='records')
    # Extraindo o nome do estado
    df_pib['estado'] = df_pib['localidade'].apply(lambda row: row['nome'])
    # Extraindo os anos
    for ano in ULTIMOS_ANOS:
        df_pib[ano] = df_pib['serie'].apply(lambda row: pd.to_numeric(row.get(str(ano), float('nan'))) * 100)
    # Remapeando os nomes dos estados para siglas
    df_pib['SIGLA'] = df_pib['estado'].map(SIGLAS_ESTADOS)
    # Removendo colunas desnecessárias
    df_pib = df_pib.drop(columns=['localidade', 'serie', 'estado'])
    # Removendo colunas com valores NaN
    df_pib = df_pib.dropna(axis=1)
    # Transpondo Anos para coluna ano
    df_pib = df_pib.melt(id_vars=['SIGLA'], var_name='ANO', value_name='VALOR')
    # Ano para inteiros
    df_pib['ANO'] = df_pib['ANO'].astype(int)
    return df_pib


def obter_dados_pib():
    """
    Obtém os dados de PIB a partir da API do IBGE.

    Funcionalidade:
        - Constrói a URL da API com base nos anos definidos em `ULTIMOS_ANOS`.
        - Realiza uma requisição HTTP para obter os dados de PIB.
        - Extrai e retorna os dados relevantes da resposta JSON.

    Retorno:
        list: Lista de dicionários contendo os dados de PIB no formato retornado pela API.

    Dependências:
        - Biblioteca `requests` para realizar a requisição HTTP.
        - A API do IBGE deve estar acessível e retornar os dados no formato esperado.

    Exceções:
        - Certifique-se de que a API está disponível e que a resposta contém os campos esperados.
        - Pode gerar exceções relacionadas à conexão ou ao formato da resposta JSON.
    """
    anos_pipe = '|'.join(map(str, ULTIMOS_ANOS))
    url_pib = f'https://servicodados.ibge.gov.br/api/v3/agregados/5938/periodos/{anos_pipe}/variaveis/37?localidades=N3[all]'
    response_pib = requests.get(url_pib)
    data_pib = response_pib.json()[0]['resultados'][0]['series']
    return data_pib


def carregar_dados_pib(df_pib):
    """
    Carrega os dados de PIB em uma tabela no banco de dados PostgreSQL.

    Parâmetros:
        df_pib (pd.DataFrame): DataFrame contendo as colunas 'SIGLA', 'ANO' e 'VALOR',
                               representando a sigla do estado, o ano e o valor do PIB.

    Funcionalidade:
        - Conecta-se ao banco de dados PostgreSQL utilizando as credenciais configuradas.
        - Cria a tabela `tabela_pib` caso ela não exista.
        - Insere os dados do DataFrame na tabela, atualizando os valores em caso de conflito.

    Dependências:
        - Biblioteca `psycopg2` para conexão com o banco de dados.
        - As variáveis globais `db_name`, `db_user`, `db_password`, `db_host` e `db_port` devem estar configuradas.

    Exceções:
        - Gera uma mensagem de erro caso ocorra algum problema na conexão ou na execução das operações no banco.

    Saída:
        - Dados inseridos ou atualizados na tabela `tabela_pib` no banco de dados PostgreSQL.
    """
    try:
        # Conexão com o banco de dados PostgreSQL
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        cursor = conn.cursor()

        # Criação da tabela se não existir
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tabela_pib (
                sigla VARCHAR(2),
                ano INT,
                valor NUMERIC,
                PRIMARY KEY (sigla, ano)
            );
        """)

        # Inserção dos dados no banco
        for _, row in df_pib.iterrows():
            cursor.execute("""
                INSERT INTO tabela_pib (sigla, ano, valor)
                VALUES (%s, %s, %s)
                ON CONFLICT (sigla, ano) DO UPDATE
                SET valor = EXCLUDED.valor;
            """, (row['SIGLA'], row['ANO'], row['VALOR']))

        # Commit e fechamento da conexão
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Erro ao carregar dados no PostgreSQL: {e}")


def carrega_dados_populacao(df_pop):
    """
    Carrega os dados de população em uma tabela no banco de dados PostgreSQL.

    Parâmetros:
        df_pop (pd.DataFrame): DataFrame contendo as colunas 'SIGLA', 'ANO' e 'VALOR',
                               representando a sigla do estado, o ano e o número de pessoas.

    Funcionalidade:
        - Conecta-se ao banco de dados PostgreSQL utilizando as credenciais configuradas.
        - Cria a tabela `tabela_pop` caso ela não exista.
        - Insere os dados do DataFrame na tabela, atualizando os valores em caso de conflito.

    Dependências:
        - Biblioteca `psycopg2` para conexão com o banco de dados.
        - As variáveis globais `db_name`, `db_user`, `db_password`, `db_host` e `db_port` devem estar configuradas.

    Exceções:
        - Gera uma mensagem de erro caso ocorra algum problema na conexão ou na execução das operações no banco.

    Saída:
        - Dados inseridos ou atualizados na tabela `tabela_pop` no banco de dados PostgreSQL.
    """
    try:
        # Conexão com o banco de dados PostgreSQL
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        cursor = conn.cursor()

        # Criação da tabela se não existir
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tabela_pop (
                sigla VARCHAR(2),
                ano INT,
                valor NUMERIC,
                PRIMARY KEY (sigla, ano)
            );
        """)

        # Inserção dos dados no banco
        for _, row in df_pop.iterrows():
            cursor.execute("""
                INSERT INTO tabela_pop (sigla, ano, valor)
                VALUES (%s, %s, %s)
                ON CONFLICT (sigla, ano) DO UPDATE
                SET valor = EXCLUDED.valor;
            """, (row['SIGLA'], row['ANO'], row['VALOR']))

        # Commit e fechamento da conexão
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Erro ao carregar dados no PostgreSQL: {e}")


def carrega_dados_pib_per_capta(tabela_pib_per_capta):
    """
    Carrega os dados de PIB per capita em uma tabela no banco de dados PostgreSQL.

    Parâmetros:
        tabela_pib_per_capta (pd.DataFrame): DataFrame contendo as colunas 'SIGLA', 'ANO' e 'PIB_PER_CAPTA',
                                             representando a sigla do estado, o ano e o valor do PIB per capita.

    Funcionalidade:
        - Conecta-se ao banco de dados PostgreSQL utilizando as credenciais configuradas.
        - Cria a tabela `tabela_pib_per_capta` caso ela não exista.
        - Insere os dados do DataFrame na tabela, atualizando os valores em caso de conflito.

    Dependências:
        - Biblioteca `psycopg2` para conexão com o banco de dados.
        - As variáveis globais `db_name`, `db_user`, `db_password`, `db_host` e `db_port` devem estar configuradas.

    Exceções:
        - Gera uma mensagem de erro caso ocorra algum problema na conexão ou na execução das operações no banco.

    Saída:
        - Dados inseridos ou atualizados na tabela `tabela_pib_per_capta` no banco de dados PostgreSQL.
    """
    try:
        # Conexão com o banco de dados PostgreSQL
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        cursor = conn.cursor()

        # Criação da tabela se não existir
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tabela_pib_per_capta (
                sigla VARCHAR(2),
                ano INT,
                valor NUMERIC,
                PRIMARY KEY (sigla, ano)
            );
        """)

        # Inserção dos dados no banco
        for _, row in tabela_pib_per_capta.iterrows():
            cursor.execute("""
                INSERT INTO tabela_pib_per_capta (sigla, ano, valor)
                VALUES (%s, %s, %s)
                ON CONFLICT (sigla, ano) DO UPDATE
                SET valor = EXCLUDED.valor;
            """, (row['SIGLA'], row['ANO'], row['PIB_PER_CAPTA']))

        # Commit e fechamento da conexão
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Erro ao carregar dados no PostgreSQL: {e}")


if __name__ == '__main__':
    main()
