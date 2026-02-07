import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('https://raw.githubusercontent.com/JohnEric-Creator/FreeDataScienceAcquisition/refs/heads/main/Titanic-Dataset.csv')
print(df)

# Data cleaning de NaN
# Ajustando valores NaN da coluna Age com fillna
# df['Age'].fillna(df['Age'].median(), inplace=True)
df['Age'] = df['Age'].fillna(df['Age'].median())

# Data cleaning de NaN
# Dropar as linhas (registros) que tem NaN nas colunas Fare ou Embarked
df = df.dropna(subset=['Fare', 'Embarked'])

# Engenharia de atributos (features engineering)
# Características derivadas
df['Family'] = df['SibSp'] + df['Parch'] + 1

# Engenharia de atributos (features engineering)
# Criação de variáveis dummies (one-hot enconding) da coluna Sex
df['Sex'] = df['Sex'].map({'male' : 1, 'female' : 0})

# Engenharia de atributos (features engineering)
# Criação de variáveis dummies (one-hot enconding) da coluna Embarked
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True, dtype=int)
df = pd.get_dummies(df, columns=['Pclass'], drop_first=True, dtype=int)
# df['Embarked_Q'] = df['Embarked_Q'].map({True : 1, False : 0})
# df['Embarked_S'] = df['Embarked_S'].map({True : 1, False : 0})

print(df['Age'].describe())

# Engenharia de atributos (features engineering)
# Discretização de variáveis contínuas (bins / intervalos) da coluna Age
bins = [0, 12, 18, 50, 100]
labels = ['Criança', 'Adolescente', 'Adulto', 'Idoso']
df['faixaEtaria'] = pd.cut(x=df['Age'], bins=bins, labels=labels)
df = pd.get_dummies(df, columns=['faixaEtaria'], drop_first=True, dtype=int)

# Engenharia de atributos (features engineering)
# Discretização de variáveis contínuas (bins / intervalos) da coluna Fare
bins = [0, 7.91, 14.54, 31, 512]
labels= ['Low', 'Mid-low', 'Mid-high', 'High']
df['faixaTarifas'] = pd.cut(x=df['Fare'], bins=bins, labels=labels)
df = pd.get_dummies(df, columns=['faixaTarifas'], drop_first=True, dtype=int)

print(df.columns)

# Separar em conjuntos de entrada e saída (target ou alvo)
# Separar em varíaveis independentes (X) e dependentes (y ou target ou alvo)
X = df[['Sex', 'Family', 'Embarked_Q', 'Embarked_S', 'Pclass_2', 'Pclass_3',
        'faixaEtaria_Adolescente', 'faixaEtaria_Adulto', 'faixaEtaria_Idoso',
        'faixaTarifas_Mid-low', 'faixaTarifas_Mid-high', 'faixaTarifas_High']]
y = df['Survived']

# Separar os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Modelo de KNN para o dataset do Titanic
model = KNeighborsClassifier(n_neighbors=15, metric='euclidean') # n_neighbors == k
model.fit(X_train, y_train)

# Previsão dos dados do dataset do Titanic
y_pred = model.predict(X_test)
print(y_pred)

# Métricas de avaliação do modelo
acuracia = accuracy_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Acurácia: {acuracia}')
print(f'F1-Score: {f1_score}')
print(f'Relatório:\n {report}')

# Matrix de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(conf_matrix, display_labels=model.classes_)
display.plot(cmap=plt.cm.Blues, values_format='g')
plt.show()

# Meus dados
print(X_test.columns)

# Meus dados
my_dict = {
    'Sex' : 1,
    'Family' : 1,
    'Embarked_Q' : 0,
    'Embarked_S' : 0,
    'Pclass_2' : 0,
    'Pclass_3' : 0,
    'faixaEtaria_Adolescente' : 0,
    'faixaEtaria_Adulto' : 1,
    'faixaEtaria_Idoso' : 0,
    'faixaTarifas_Mid-low' : 0,
    'faixaTarifas_Mid-high' : 0,
    'faixaTarifas_High' : 1
}

nome = ['André']
my_test = pd.DataFrame(my_dict, index=nome)
print(my_test)

# Previsão individual
y_pred = model.predict(my_test)
print(f'Sobreviveu? {y_pred}')