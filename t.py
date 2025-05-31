import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# Wczytaj dane
df = pd.read_csv("top10K-TMDB-movies.csv")

# Folder zapisu
output_folder = "eda_plots"
os.makedirs(output_folder, exist_ok=True)

# --- Informacje podstawowe ---
print("Podstawowe informacje o zbiorze danych:")
print(df.info())
print("\nOpis statystyczny danych liczbowych:")
print(df.describe())
print("\nBrakujące wartości:")
print(df.isnull().sum())

# --- Przekształcanie dat ---
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year

# --- Analiza kolumn ---
df['genre'] = df['genre'].fillna('')
df['genre_list'] = df['genre'].apply(lambda x: [g.strip() for g in x.split(',') if g.strip()])
df['description_length'] = df['overview'].fillna('').apply(lambda x: len(x.split()))

# --- Gatunki ---
genre_counter = Counter([genre for sublist in df['genre_list'] for genre in sublist])
common_genres = pd.DataFrame(genre_counter.items(), columns=['Genre', 'Count']).sort_values(by='Count', ascending=False)

# --- Wykres 1: Gatunki ---
plt.figure(figsize=(10, 6))
common_genres.head(10).plot(kind='bar', x='Genre', y='Count', legend=False)
plt.title("Top 10 najczęstszych gatunków filmowych")
plt.ylabel("Liczba filmów")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_folder}/genre_barplot.png")
plt.close()

# --- Wykres 2: Rozkład ocen ---
plt.figure(figsize=(10, 6))
sns.histplot(df['vote_average'].dropna(), bins=20, kde=True)
plt.title("Rozkład średnich ocen filmów")
plt.xlabel("Średnia ocena")
plt.ylabel("Liczba filmów")
plt.tight_layout()
plt.savefig(f"{output_folder}/ratings_hist.png")
plt.close()

# --- Wykres 3: Liczba głosów (log Y) ---
plt.figure(figsize=(10, 6))
sns.histplot(df['vote_count'].dropna(), bins=50, log_scale=(False, True))
plt.title("Rozkład liczby głosów (logarytmiczna oś Y)")
plt.xlabel("Liczba głosów")
plt.tight_layout()
plt.savefig(f"{output_folder}/vote_count_hist.png")
plt.close()

# --- Wykres 4: Zależność liczby głosów i ocen ---
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='vote_count', y='vote_average')
plt.title("Zależność: liczba głosów vs średnia ocena")
plt.xlabel("Liczba głosów")
plt.ylabel("Średnia ocena")
plt.tight_layout()
plt.savefig(f"{output_folder}/votes_vs_ratings.png")
plt.close()

# --- Wykres 5: Filmy na przestrzeni lat ---
plt.figure(figsize=(12, 6))
df['release_year'].value_counts().sort_index().plot()
plt.title("Liczba filmów wydanych w poszczególnych latach")
plt.xlabel("Rok")
plt.ylabel("Liczba filmów")
plt.tight_layout()
plt.savefig(f"{output_folder}/release_years.png")
plt.close()

# --- Wykres 6: Korelacje ---
num_cols = ['popularity', 'vote_average', 'vote_count', 'description_length']
plt.figure(figsize=(8, 6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Macierz korelacji cech liczbowych")
plt.tight_layout()
plt.savefig(f"{output_folder}/correlation_matrix.png")
plt.close()

# --- Wykres 7: Liczba filmów w danym języku ---
plt.figure(figsize=(12, 6))
lang_counts = df['original_language'].value_counts().head(15)  # najczęstsze 15 języków
lang_counts.plot(kind='bar')
plt.title("Liczba filmów według języka oryginalnego (Top 15)")
plt.xlabel("Język oryginalny")
plt.ylabel("Liczba filmów")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_folder}/language_distribution.png")
plt.close()

print(f"Wszystkie wykresy zostały zapisane w folderze: {output_folder}")
