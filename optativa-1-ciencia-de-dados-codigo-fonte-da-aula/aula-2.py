#eda aula-2

import pandas as pd

df = pd.read_csv('data.csv')
df

df.sort_values(by=["Duration", "Calories"], ascending=False, inplace=False, ignore_index=True)

# Head and tail
# df.head(3)
# df.tail(5)

df.dropna(axis=0, how='any', inplace=True)
df

df.loc[7, 'Duration'] = 45

df.loc[26, 'Date'] = "'2020/12/26'"

# Removendo colunas (características)
colunas = ['Duration', 'Pulse', 'Maxpulse', 'Calories']
df = df[colunas]
df

# Correlação de variáveis (características)
df.corr()

# Shape (linhas, colunas)
df.shape

# Columns (colunas)
df.columns


# Info
df.info()

# Describe
df.describe()

# Value Counts
df.value_counts()

# Medidas estatísticas
df['Calories'].mean()

Análise Exploratória Inicial

EDA

- Limpeza de dados
- Células vazias
- Dados em formato errado
- Dados duplicados
