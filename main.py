import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv('titanic.csv')
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df['Embarked'].fillna('S', inplace=True)
age_1 =  df[df['Pclass'] == 1]['Age'].median()
age_2 =  df[df['Pclass'] == 2]['Age'].median()
age_3 =  df[df['Pclass'] == 3]['Age'].median()

def fill_age(row):
    if pd.isnull(row['Age']):
        if row['Pclass'] == 1:
            return age_1
        if row['Pclass'] == 2:
            return age_2
        return age_3
    return row['Age']

df['Age'] = df.apply(fill_age, axis=1)
df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
embarked_df = pd.get_dummies(df['Embarked'])
df.drop('Embarked', inplace=True, axis=1)
df = df.join(embarked_df, how='right')
# df = pd.concat([df, embarked_df], axis=1)

x = df.drop('Survived', axis=1)
y = df['Survived']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

print(x_train.describe())
sc = StandardScaler()
print(f'До Z оценки: {x_train.iloc[0, :3]}')
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(f'После Z оценки: {x_train[0, :3]}')

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print(f'Процент правильно предсказанных исходов: {accuracy_score(y_pred, y_test) * 100}')
print(confusion_matrix(y_pred, y_test))