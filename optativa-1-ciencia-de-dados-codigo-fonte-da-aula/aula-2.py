# Open In Colab

# Importação
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report ,ConfusionMatrixDisplay


# Tabela de idade , altura, peso, salario, faculdade, nome, time
df= pd.read_csv('https://media.geeksforgeeks.org/wp-content/uploads/nba.csv')
df
#
# Name 	Team 	Number 	Position 	Age 	Height 	Weight 	College 	Salary
# 0 	Avery Bradley 	Boston Celtics 	0.0 	PG 	25.0  6- 2
# 180.0
# Texas
# 7730337.0
# 1
# Jae
# Crowder
# Boston
# Celtics
# 99.0
# SF
# 25.0
# 6 - 6
# 235.0
# Marquette
# 6796117.0
# 2
# John
# Holland
# Boston
# Celtics
# 30.0
# SG
# 27.0
# 6 - 5
# 205.0
# Boston
# University
# NaN
# 3
# R.J.Hunter
# Boston
# Celtics
# 28.0
# SG
# 22.0
# 6 - 5
# 185.0
# Georgia
# State
# 1148640.0
# 4
# Jonas
# Jerebko
# Boston
# Celtics
# 8.0
# PF
# 29.0
# 6 - 10
# 231.0
# NaN
# 5000000.0
# ... ... 	... 	... 	... 	... 	... 	... 	... 	...
# 453 	Shelvin Mack 	Utah Jazz 	8.0 	PG 	26.0  6- 3
# 203.0
# Butler
# 2433333.0
# 454
# Raul
# Neto
# Utah
# Jazz
# 25.0
# PG
# 24.0
# 6 - 1
# 179.0
# NaN
# 900000.0
# 455
# Tibor
# Pleiss
# Utah
# Jazz
# 21.0
# C
# 26.0
# 7 - 3
# 256.0
# NaN
# 2900000.0
# 456
# Jeff
# Withey
# Utah
# Jazz
# 24.0
# C
# 26.0
# 7 - 0
# 231.0
# Kansas
# 947276.0
# 457
# NaN
# NaN
# NaN
# NaN
# NaN
# NaN
# NaN
# NaN
# NaN
#
# 458
# rows × 9
# columns

df['Age'] = df['Age'] + df['Weight'] + 1

df.drop(columns='Salary')

# Name
# Team
# Number
# Position
# Age
# Height
# Weight
# College
# 0
# Avery
# Bradley
# Boston
# Celtics
# 0.0
# PG
# 206.0
# 6 - 2
# 180.0
# Texas
# 1
# Jae
# Crowder
# Boston
# Celtics
# 99.0
# SF
# 261.0
# 6 - 6
# 235.0
# Marquette
# 2
# John
# Holland
# Boston
# Celtics
# 30.0
# SG
# 233.0
# 6 - 5
# 205.0
# Boston
# University
# 3
# R.J.Hunter
# Boston
# Celtics
# 28.0
# SG
# 208.0
# 6 - 5
# 185.0
# Georgia
# State
# 4
# Jonas
# Jerebko
# Boston
# Celtics
# 8.0
# PF
# 261.0
# 6 - 10
# 231.0
# NaN
# ... ... 	... 	... 	... 	... 	... 	... 	...
# 453 	Shelvin Mack 	Utah Jazz 	8.0 	PG 	230.0  6- 3
# 203.0
# Butler
# 454
# Raul
# Neto
# Utah
# Jazz
# 25.0
# PG
# 204.0
# 6 - 1
# 179.0
# NaN
# 455
# Tibor
# Pleiss
# Utah
# Jazz
# 21.0
# C
# 283.0
# 7 - 3
# 256.0
# NaN
# 456
# Jeff
# Withey
# Utah
# Jazz
# 24.0
# C
# 258.0
# 7 - 0
# 231.0
# Kansas
# 457
# NaN
# NaN
# NaN
# NaN
# NaN
# NaN
# NaN
# NaN
#
# 458
# rows × 8
# columns

df.dropna(subset='Height', inplace=True)
df.dropna(subset='Weight', inplace=True)

df = df.drop_duplicates()


def convertWeight(x):
    return x * 0.453592


texto = '175 cm'
lista = texto.split()
lista

['175', 'cm']


def convertHeight(x):
    foot = int(x.split('-')[0])
    inch = int(x.split('-')[1])
    return (foot * 12 + inch) * 0.0254  # convertido tudo para metro


# def arvore():


# compreensão
df['Weight(kg)'] = [convertWeight(x) for x in df['Weight']]
df['Height(m)'] = [convertHeight(x) for x in df['Height']]

colunas = ['Name', 'Team', 'Position', 'Age', 'Height(m)', 'Weight(kg)', 'College', 'Salary']
df = df[colunas]

df

# Name
# Team
# Position
# Age
# Height(m)
# Weight(kg)
# College
# Salary
# 0
# Avery
# Bradley
# Boston
# Celtics
# PG
# 206.0
# 1.8796
# 81.646560
# Texas
# 7730337.0
# 1
# Jae
# Crowder
# Boston
# Celtics
# SF
# 261.0
# 1.9812
# 106.594120
# Marquette
# 6796117.0
# 2
# John
# Holland
# Boston
# Celtics
# SG
# 233.0
# 1.9558
# 92.986360
# Boston
# University
# NaN
# 3
# R.J.Hunter
# Boston
# Celtics
# SG
# 208.0
# 1.9558
# 83.914520
# Georgia
# State
# 1148640.0
# 4
# Jonas
# Jerebko
# Boston
# Celtics
# PF
# 261.0
# 2.0828
# 104.779752
# NaN
# 5000000.0
# ... ... 	... 	... 	... 	... 	... 	... 	...
# 452 	Trey Lyles 	Utah Jazz 	PF 	255.0 	2.0828 	106.140528 	Kentucky 	2239800.0
# 453 	Shelvin Mack 	Utah Jazz 	PG 	230.0 	1.9050 	92.079176 	Butler 	2433333.0
# 454 	Raul Neto 	Utah Jazz 	PG 	204.0 	1.8542 	81.192968 	NaN 	900000.0
# 455 	Tibor Pleiss 	Utah Jazz 	C 	283.0 	2.2098 	116.119552 	NaN 	2900000.0
# 456 	Jeff Withey 	Utah Jazz 	C 	258.0 	2.1336 	104.779752 	Kansas 	947276.0
#
# 457 rows × 8 columns

# Groupby
grupos = df.groupby(by='Position')
grupos.first()

# Name 	Team 	Age 	Height(m) 	Weight(kg) 	College 	Salary
# Position
# C 	Kelly Olynyk 	Boston Celtics 	264.0 	2.1336 	107.954896 	Gonzaga 	2165160.0
# PF 	Jonas Jerebko 	Boston Celtics 	261.0 	2.0828 	104.779752 	LSU 	5000000.0
# PG 	Avery Bradley 	Boston Celtics 	206.0 	1.8796 	81.646560 	Texas 	7730337.0
# SF 	Jae Crowder 	Boston Celtics 	261.0 	1.9812 	106.594120 	Marquette 	6796117.0
# SG 	John Holland 	Boston Celtics 	233.0 	1.9558 	92.986360 	Boston University 	1148640.0

# Grupos
grupos.get_group('SF')['Age'].mean()


np.float64(249.63529411764705)


# Grupos
# grupos.get_group('Boston Celtics')['Age'].median()
# grupos.get_group('Boston Celtics')['Age'].mean()
grupos.get_group('SG')['Age'].mean()



np.float64(234.22549019607843)


df.columns


Index(['Name', 'Team', 'Position', 'Age', 'Height(m)', 'Weight(kg)', 'College',
       'Salary'],
      dtype='object')


df.info()
#
# <class 'pandas.core.frame.DataFrame'>
# I


# # dex: 457 e
# # tries, 0 to 4
# # 6
# # Data c
# # lumns (total 8
# # columns):
# # #   Column      Non-Null Count  Dtype
# # ---  ------      --------------  -----
# # 0
# # Name
# # 457
# # non- n ull
# # object
# # 1
# # Team
# # 457
# # non- n ull
# # object
# # 2
# # Position
# # 457
# # non- n ull
# # object
# # 3
# # Age
# # 457
# # non- n ull
# # float64
# # 4
# # Height(m)
# # 457
# # non- n ull
# # float64
# 5
# Weight(kg)
# 457
# non- n ull
# float64
# 6
# College
# 373
# non- n ull
# object
# 7
# Salary
# 446
# non- n ull
# float64
# dtypes: float64(4), object(4)
# memory
# usage: 32.1+ KB

df.describe()

# Age
# Height(m)
# Weight(kg)
# Salary
# count
# 457.000000
# 457.000000
# 457.000000
# 4.460000e+02
# mean
# 249.461707
# 2.011435
# 100.481050
# 4.842684e+06
# std
# 27.109653
# 0.087184
# 11.960469
# 5.229238e+06
# min
# 189.000000
# 1.752600
# 73.028312
# 3.088800e+04
# 25 % 228.000000
# 1.955800
# 90.718400
# 1.044792e+06
# 50 % 249.000000
# 2.032000
# 99.790240
# 2.839073e+06
# 75 % 270.000000
# 2.082800
# 108.862080
# 6.500000e+06
# max
# 338.000000
# 2.209800
# 139.252744
# 2.500000e+07

GroupBy = df.groupby(by='Salary')
GroupBy.first()

# Name
# Team
# Position
# Age
# Height(m)
# Weight(kg)
# College
# Salary
# 30888.0
# Thanasis
# Antetokounmpo
# New
# York
# Knicks
# SF
# 229.0
# 2.0066
# 92.986360
# None
# 55722.0
# Phil
# Pressey
# Phoenix
# Suns
# PG
# 201.0
# 1.8034
# 79.378600
# Missouri
# 83397.0
# Alan
# Williams
# Phoenix
# Suns
# C
# 284.0
# 2.0320
# 117.933920
# UC
# Santa
# Barbara
# 111196.0
# Jordan
# McRae
# Cleveland
# Cavaliers
# SG
# 205.0
# 1.9558
# 81.192968
# Tennessee
# 111444.0
# Jeff
# Ayres
# Los
# Angeles
# Clippers
# PF
# 280.0
# 2.0574
# 113.398000
# Arizona
# State
# ... ... 	... 	... 	... 	... 	... 	...
# 22192730.0 	Chris Bosh 	Miami Heat 	PF 	268.0 	2.1082 	106.594120 	Georgia Tech
# 22359364.0 	Dwight Howard 	Houston Rockets 	C 	296.0 	2.1082 	120.201880 	None
# 22875000.0 	Carmelo Anthony 	New York Knicks 	SF 	273.0 	2.0320 	108.862080 	Syracuse
# 22970500.0 	LeBron James 	Cleveland Cavaliers 	SF 	282.0 	2.0320 	113.398000 	None
# 25000000.0 	Kobe Bryant 	Los Angeles Lakers 	SF 	250.0 	1.9812 	96.161504 	None
#
# 309 rows × 7 columns

f1 = f1_score(y_test, y_pred, average='weighted')
acuracia = accuracy_score(y_test, y_pred)
reporta = classification_report(y_test, y_pred)

print(f"Relatório:\n{reporta}")
print(f"Acurácia: {acuracia}")
print(f"F1 Score: {f1}")


# Relatório:
# precision    recall  f 1 -score   support
#
# C       0.73      0.79      0.76        14
# PF       0.41      0.39      0.40        18
# PG       0.83      0.75      0.79        20
# SF       0.37      0.41      0.39        17
# SG       0.57      0.57      0.57        23
#
# accuracy                           0.58        92
# macro avg       0.58      0.58      0.58        92
# weighted avg       0.58      0.58      0.58        92

# Acurácia: 0.5760869565217391
# F1 Score: 0.5784921135923267

