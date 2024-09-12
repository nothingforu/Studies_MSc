import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Wczytaj oba pliki CSV
df1 = pd.read_csv('gender_submission.csv')
df2 = pd.read_csv('test.csv')

df3=pd.read_csv('train.csv')
max_age1 = df3['Fare'].max()
max_age = df2['Fare'].max()
print(f'Maksymalna wartość w kolumnie "Fare": {max_age}')
print(f'Maksymalna wartość w kolumnie "Fare_train": {max_age1}')
# Wybierz odpowiednie kolumny do połączenia
df1_selected = df1[['PassengerId', 'Survived']]
df2_selected = df2[['PassengerId', 'Sex', 'Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked']]

# Wypełnij brakujące wartości w kolumnie 'Embarked' najczęściej występującą wartością
most_common_embarked = df2['Embarked'].mode()[0]
df2['Embarked'] = df2['Embarked'].fillna(most_common_embarked)

# Zamień wartości 'S', 'C', 'Q' na liczby 0, 1, 2
df2['Embarked'] = df2['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Zapisz zmodyfikowaną kolumnę 'Embarked' do pliku test.csv
df2_selected['Embarked'] = df2['Embarked']  # Upewnij się, że df2_selected zawiera zaktualizowaną kolumnę

# Połącz dane na podstawie 'PassengerId'
combined_df = pd.merge(df1_selected, df2_selected, on='PassengerId', how='inner')

# Zapisz wynik do nowego pliku CSV z odpowiednimi wartościami Embarked (liczbowymi)
combined_df.to_csv('polaczony_plik.csv', index=False)

# Wczytaj dane z pliku CSV, aby je zwizualizować
existing_df = pd.read_csv('połączony_plik.csv')

# Wykres - liczba osób, które przeżyły w zależności od płci
plt.figure(figsize=(10, 7))
sns.countplot(x='Sex', hue='Survived', data=existing_df)
plt.title('Przeżywalność w zależności od płci')

plt.tight_layout()
plt.show()
