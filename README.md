# Autores
- Alisson Santos da Silveira
- Felipe Pereira de Castro
- Gabriel Sperb Stoffel
- Karina Murta Starling Hayashi

# Projeto de Análise de PIB e População por Estado

Este projeto realiza a análise de dados de PIB e população dos estados brasileiros, gerando gráficos comparativos e previsões de PIB per capita. Os dados são obtidos de fontes externas, transformados e armazenados em um banco de dados PostgreSQL.

## Funcionalidades

- **Obtenção de Dados**:
  - PIB: Obtido via API do IBGE no formato **JSON** (semi-estruturados).
  - População: Arquivo Excel (**tabela - CSV - estruturado**) baixado de um link do Google Drive.

- **Transformação de Dados**:
  - Dados de PIB e população são processados para incluir apenas os anos e estados relevantes.
  - Cálculo do PIB per capita com base nos dados de PIB e população.

- **Armazenamento de Dados**:
  - Os dados são armazenados em tabelas no banco de dados PostgreSQL:
    - `tabela_pib`: Dados de PIB.
    - `tabela_pop`: Dados de população.
    - `tabela_pib_per_capta`: Dados de PIB per capita.

- **Geração de Gráficos**:
  - Gráficos comparativos de PIB, população e PIB per capita por estado e ano.
  - Previsões de PIB per capita até 2025 utilizando regressão linear.

## Estrutura do Projeto

- `main.py`: Arquivo principal que executa o fluxo completo do projeto.
- `requirements.txt`: Dependências do projeto.
- `data/`: Diretório onde o arquivo Excel de população é salvo.
- `output/`: Diretório onde os gráficos gerados são salvos.

## Execução
A execução do projeto é através do docker-compose, que cria um container com o banco de dados PostgreSQL e executa o script Python para realizar a análise.

Para inicializar o projeto, siga os passos abaixo:
```bash
docker-compose up
```

Para parar o container, utilize:
```bash
docker-compose down
```

## Dependências
- Python 3.11 ou superior
- PostgreSQL
- Bibliotecas Python:
  - Pandas
  - Matplotlib
  - NumPy
  - Scikit-learn
  - Requests
  - SQLAlchemy
  - Psycopg2
  - Openpyxl
  - Seaborn
