import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import pickle


df = pd.read_csv('train.csv')

sns.barplot(x='Sex', y='Survived', data=df)
#plt.show()

b = int(df.Age.median())
df.Age = df.Age.fillna(b)

le_gender = LabelEncoder()

df['Gender'] = le_gender.fit_transform(df['Sex'])
#print(df['Gender'])

target = df.Survived
input = df.drop(['PassengerId', 'Survived', 'Pclass', 'Name', 'Ticket', 'Fare',	'Cabin', 'Embarked', 'Sex'], axis=1)

model = tree.DecisionTreeClassifier()
model.fit(input, target)

print(model.score(input.tail(), target.tail()))

with open('Titanic_Survival', 'wb') as f:
    pickle.dump('model', f)

