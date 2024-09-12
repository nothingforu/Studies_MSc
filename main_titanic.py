import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
import numpy as np

# Ustawienie, aby wyświetlić wszystkie kolumny
pd.set_option('display.max_columns', None)

# Wczytywanie danych
train_df = pd.read_csv('train.csv')
print("Brakujące wartości w danych treningowych:\n", train_df.isnull().sum())

# Uzupełnienie brakujących wartości
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
most_common_embarked = train_df['Embarked'].mode()[0]
train_df['Embarked'] = train_df['Embarked'].fillna(most_common_embarked)

print(train_df.describe())

# Ignorowanie kolumny 'Cabin', ponieważ ma zbyt wiele braków
train_df.drop(columns=['Cabin'], inplace=True)
train_df.drop(columns=['Ticket'], inplace=True)

# Konwersja zmiennych kategorii do wartości liczbowych
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Wybór cech do modelu
features = ['Pclass', 'Age', 'Fare', 'Sex', 'SibSp', 'Parch', 'Embarked']
X = train_df[features]
y = train_df['Survived']


# Budowa i ocena modelu RandomForest
n_estimators_range = range(10, 121, 10)
cv_scores_rf = []
cv_f1_scores_rf = []
# Wybranie najbardziej optymalnej ilosci drzew dla RF
for n_estimators in n_estimators_range:
    model_rf = RandomForestClassifier(n_estimators=n_estimators, criterion="log_loss")
    scores = cross_val_score(model_rf, X, y, cv=10, scoring='accuracy')
    f1_scores = cross_val_score(model_rf, X, y, cv=10, scoring='f1')
    cv_scores_rf.append(scores.mean())
    cv_f1_scores_rf.append(f1_scores.mean())

optimal_n_estimators = n_estimators_range[np.argmax(cv_scores_rf)]
model_rf = RandomForestClassifier(n_estimators=optimal_n_estimators, criterion="log_loss")
model_rf.fit(X, y)
scores_rf = cross_val_score(model_rf, X, y, cv=10, scoring='accuracy')
f1_scores_rf = cross_val_score(model_rf, X, y, cv=10, scoring='f1')

print(f'Optymalna liczba drzew dla RandomForest: {optimal_n_estimators}')
print(f'Średnia dokładność modelu RandomForest (CV): {scores_rf.mean() * 100:.2f}%')
print(f'Średni F1 score modelu RandomForest (CV): {f1_scores_rf.mean() * 100:.2f}%')

# Budowa i ocena modelu KNN
neighbors_range = range(1, 11)
cv_scores_knn = []
cv_f1_scores_knn = []
# Wybranie najbardziej optymalnej ilosci sąsiadów dla KNN
for n_neighbors in neighbors_range:
    model_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(model_knn, X, y, cv=10, scoring='accuracy')
    f1_scores = cross_val_score(model_knn, X, y, cv=10, scoring='f1')
    cv_scores_knn.append(scores.mean())
    cv_f1_scores_knn.append(f1_scores.mean())

optimal_n_neighbors = neighbors_range[np.argmax(cv_scores_knn)]
model_knn = KNeighborsClassifier(n_neighbors=optimal_n_neighbors)
model_knn.fit(X, y)
scores_knn = cross_val_score(model_knn, X, y, cv=10, scoring='accuracy')
f1_scores_knn = cross_val_score(model_knn, X, y, cv=10, scoring='f1')

print(f'Optymalna liczba sąsiadów dla KNN: {optimal_n_neighbors}')
print(f'Średnia dokładność modelu KNN (CV): {scores_knn.mean() * 100:.2f}%')
print(f'Średni F1 score modelu KNN (CV): {f1_scores_knn.mean() * 100:.2f}%')

# Wykres dokładności i F1 score dla KNN w zależności od liczby sąsiadów
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(neighbors_range, cv_scores_knn, marker='o', label='Dokładność')
plt.title('Dokładność KNN w zależności od liczby sąsiadów')
plt.xlabel('Liczba sąsiadów (n_neighbors)')
plt.ylabel('Dokładność (Accuracy)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(neighbors_range, cv_f1_scores_knn, marker='o', label='F1 Score', color='orange')
plt.title('F1 Score KNN w zależności od liczby sąsiadów')
plt.xlabel('Liczba sąsiadów (n_neighbors)')
plt.ylabel('F1 Score')
plt.grid(True)

plt.tight_layout()
plt.show()

# Wykres dokładności i F1 score dla RandomForest w zależności od liczby drzew
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(n_estimators_range, cv_scores_rf, marker='o', label='Dokładność')
plt.title('Dokładność RandomForest w zależności od liczby drzew')
plt.xlabel('Liczba drzew (n_estimators)')
plt.ylabel('Dokładność (Accuracy)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(n_estimators_range, cv_f1_scores_rf, marker='o', label='F1 Score', color='orange')
plt.title('F1 Score RandomForest w zależności od liczby drzew')
plt.xlabel('Liczba drzew (n_estimators)')
plt.ylabel('F1 Score')
plt.grid(True)

plt.tight_layout()
plt.show()


# Zapis wyników cross-validation do plików CSV
cv_results_rf = pd.DataFrame({
    'n_estimators': n_estimators_range,
    'mean_accuracy': cv_scores_rf,
    'mean_f1_score': cv_f1_scores_rf
})
cv_results_rf.to_csv('cv_results_random_forest.csv', index=False)

cv_results_knn = pd.DataFrame({
    'n_neighbors': neighbors_range,
    'mean_accuracy': cv_scores_knn,
    'mean_f1_score': cv_f1_scores_knn
})
cv_results_knn.to_csv('cv_results_knn.csv', index=False)



# Wczytaj zbiór testowy (bez wyników przeżycia)
test_df = pd.read_csv('test.csv')

print("Brakujące wartości w danych testowych:\n", test_df.isnull().sum())
# Oczyszczenie danych w zbiorze testowym
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())
test_df['Embarked'] = test_df['Embarked'].fillna(most_common_embarked)
test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

# Wybór cech
X_new = test_df[features]

# Przewidywanie wyników z modelu KNN na zbiorze testowym
predictions_knn = model_rf.predict(X_new)
#predictions_knn = model_knn.predict(X_new)
test_df['Survived'] = predictions_knn

# Zapis wyników
output = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions_knn,
    'Pclass': test_df['Pclass'],
    'Age': test_df['Age'],
    'Sex': test_df['Sex'],
    'Fare': test_df['Fare'],
    'SibSp': test_df['SibSp'],
    'Parch': test_df['Parch'],
    'Embarked': test_df['Embarked']})
output.to_csv('my_submission_rf.csv', index=False)
#output.to_csv('my_submission_knn.csv', index=False)
existing_df = pd.read_csv('polaczony_plik.csv')
# Wczytaj i porównaj wyniki z pliku
 #połaczyłem pliki testowe oraz gender_sumbisson do pokazania wykresów
knn_df = pd.read_csv('my_submission_knn.csv')
rf_df = pd.read_csv('my_submission_rf.csv')

print("Wyniki zostały zapisane do pliku my_submission.csv")
my_submission = pd.read_csv('my_submission.csv')

plt.figure(figsize=(15, 10))

# Subplot 1
plt.subplot(2, 4, 1)
sns.countplot(x='Embarked', hue='Survived', data=existing_df)
plt.title('Embarked a przeżycie - Dane testowe')

# Subplot 2
plt.subplot(2, 4, 2)
sns.countplot(x='Embarked', hue='Survived', data=knn_df)
plt.title('Embarked a przeżycie - KNN')

# Subplot 3
plt.subplot(2, 4, 3)
sns.countplot(x='Embarked', hue='Survived', data=rf_df)
plt.title('Embarked a przeżycie - RandomForest')

# Subplot 4
plt.subplot(2, 4, 4)
sns.countplot(x='Embarked', hue='Survived', data=train_df)
plt.title('Embarked a przeżycie - Dane treningowe')

# Subplot 5
plt.subplot(2, 4, 5)
sns.countplot(x='Sex', hue='Survived', data=existing_df,dodge=True)
plt.title('Płeć a przeżycie - Dane testowe')
#plt.xticks(range(0, 91, 10))  # Ustawia wartości co 10 na osi X

# Subplot 6
plt.subplot(2, 4, 6)
sns.countplot(x='Sex', hue='Survived', data=knn_df,dodge=True)
plt.title('Płeć a przeżycie - KNN')
#plt.xticks(range(0, 91, 10))  # Ustawia wartości co 10 na osi X

# Subplot 7
plt.subplot(2, 4, 7)
sns.countplot(x='Sex', hue='Survived', data=rf_df,dodge=True)
plt.title('Płeć a przeżycie - RandomForest')
#plt.xticks(range(0, 91, 10))  # Ustawia wartości co 10 na osi X

# Subplot 7
plt.subplot(2, 4, 8)
sns.countplot(x='Sex', hue='Survived', data=train_df,dodge=True)
plt.title('Płeć a przeżycie - Dane treningowe')
#plt.xticks(range(0, 91, 10))  # Ustawia wartości co 10 na osi X

plt.tight_layout()
plt.show()

