import pandas as pd
import numpy as np
import math
from collections import defaultdict, Counter

class MovieRecommender:
    def __init__(self, file_path="top10K-TMDB-movies.csv"):
        """Initialize the recommender by reading from a CSV file."""
        # Load CSV file using pandas
        self.df = pd.read_csv(file_path)
        # Convert to list of dictionaries
        self.movies = self.df.to_dict('records')
        # Extract unique genres and build feature vectors
        self._build_feature_vectors()

    def _build_feature_vectors(self):
        """Build feature vectors for genres, normalized ratings, and rating bins."""
        # Collect all unique genres
        self.all_genres = set()
        for movie in self.movies:
            if isinstance(movie['genre'], str):
                genres = movie['genre'].split(',')
                self.all_genres.update(genres)
        self.genre_list = sorted(list(self.all_genres))  # Sorted for consistent vector order

        # Normalize vote_average (scale to [0,1]) and create rating bins
        vote_averages = [float(movie['vote_average']) if isinstance(movie['vote_average'], (int, float)) and not np.isnan(movie['vote_average']) else 0 for movie in self.movies]
        self.rating_min = min([v for v in vote_averages if v > 0], default=1)
        self.rating_max = max(vote_averages, default=10)
        self.rating_bins = [0, 4, 7, 10]  # Bins: low (0-4), medium (4-7), high (7-10)

        # Create feature vectors for each movie
        for movie in self.movies:
            # Genre vector (one-hot encoding)
            if isinstance(movie['genre'], str):
                movie_genres = set(movie['genre'].split(','))
            else:
                movie_genres = set()  # Handle NaN or non-string
            movie['genre_vector'] = [1 if genre in movie_genres else 0 for genre in self.genre_list]
            movie['genres_set'] = frozenset(movie_genres)  # For Naive Bayes class

            # Normalized rating and rating bin
            if isinstance(movie['vote_average'], (int, float)) and not np.isnan(movie['vote_average']):
                movie['norm_rating'] = (movie['vote_average'] - self.rating_min) / (self.rating_max - self.rating_min)
                # Assign rating bin (0: low, 1: medium, 2: high)
                movie['rating_bin'] = next(i for i, threshold in enumerate(self.rating_bins[1:]) if movie['vote_average'] <= threshold)
            else:
                movie['norm_rating'] = 0
                movie['rating_bin'] = 0  # Default to low bin

        # Build Naive Bayes model
        self._build_naive_bayes()

    def _build_naive_bayes(self):
        """Build Naive Bayes model for genre-based classification."""
        # Count occurrences of each genre set (class)
        self.class_counts = Counter(movie['genres_set'] for movie in self.movies)
        self.total_movies = len(self.movies)
        
        # Compute likelihoods: P(genre_i=1 | class) and P(rating_bin | class)
        self.genre_likelihoods = defaultdict(lambda: defaultdict(lambda: 1 / (self.total_movies + 2)))  # Laplace smoothing
        self.rating_likelihoods = defaultdict(lambda: defaultdict(lambda: 1 / (self.total_movies + 3)))  # Laplace smoothing
        for movie in self.movies:
            cls = movie['genres_set']
            for i, genre in enumerate(self.genre_list):
                if movie['genre_vector'][i]:
                    self.genre_likelihoods[cls][genre] += 1
            self.rating_likelihoods[cls][movie['rating_bin']] += 1

        # Normalize likelihoods
        for cls in self.class_counts:
            class_count = self.class_counts[cls] + 2  # Laplace smoothing for genres
            for genre in self.genre_list:
                self.genre_likelihoods[cls][genre] /= class_count
            class_count_rating = self.class_counts[cls] + 3  # Laplace smoothing for rating bins
            for bin_idx in range(len(self.rating_bins) - 1):
                self.rating_likelihoods[cls][bin_idx] /= class_count_rating

    def _cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two genre vectors."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a ** 2 for a in vec1))
        norm2 = math.sqrt(sum(b ** 2 for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0
        return dot / (norm1 * norm2)

    def _euclidean_distance(self, rating1, rating2):
        """Compute Euclidean distance between normalized ratings (inverted for similarity)."""
        return 1 / (1 + abs(rating1 - rating2))

    def _knn_similarity(self, movie1, movie2, genre_weight=0.7, rating_weight=0.3):
        """Compute similarity for k-NN between two movies."""
        genre_sim = self._cosine_similarity(movie1['genre_vector'], movie2['genre_vector'])
        rating_sim = self._euclidean_distance(movie1['norm_rating'], movie2['norm_rating'])
        return genre_weight * genre_sim + rating_weight * rating_sim

    def _naive_bayes_score(self, movie, target_genres_set, target_rating_bin):
        """Compute Naive Bayes score for a movie given target genres and rating bin."""
        cls = movie['genres_set']
        # Prior: P(class)
        prior = (self.class_counts[cls] + 1) / (self.total_movies + len(self.class_counts))
        
        # Likelihood: P(features | class)
        log_likelihood = 0
        for i, genre in enumerate(self.genre_list):
            feature_value = 1 if genre in target_genres_set else 0
            prob = self.genre_likelihoods[cls][genre] if feature_value else (1 - self.genre_likelihoods[cls][genre])
            log_likelihood += math.log(prob + 1e-10)  # Avoid log(0)
        
        # Rating likelihood
        prob_rating = self.rating_likelihoods[cls][target_rating_bin]
        log_likelihood += math.log(prob_rating + 1e-10)
        
        return prior * math.exp(log_likelihood)

    def recommend(self, title, k=5, method="knn"):
        """Recommend k movies similar to the given title using specified method."""
        # Find the target movie (case-insensitive)
        target_movie = None
        for movie in self.movies:
            if isinstance(movie['title'], str) and movie['title'].lower() == title.lower():
                target_movie = movie
                break
        if not target_movie:
            return []

        # Compute similarities/scores for all other movies
        scores = []
        if method.lower() == "knn":
            for movie in self.movies:
                if movie['title'] != target_movie['title']:  # Exclude the movie itself
                    sim = self._knn_similarity(target_movie, movie)
                    scores.append((sim, movie['title']))
        elif method.lower() == "naive_bayes":
            target_genres_set = target_movie['genres_set']
            target_rating_bin = target_movie['rating_bin']
            for movie in self.movies:
                if movie['title'] != target_movie['title']:  # Exclude the movie itself
                    score = self._naive_bayes_score(movie, target_genres_set, target_rating_bin)
                    scores.append((score, movie['title']))
        else:
            raise ValueError("Method must be 'knn' or 'naive_bayes'")

        # Sort by score descending and return top k titles
        scores.sort(reverse=True)
        return [title for _, title in scores[:k]]

# Example usage
if __name__ == "__main__":
    try:
        recommender = MovieRecommender("top10K-TMDB-movies.csv")
        # Test k-NN
        movie = "John Wick"
        print(f"k-NN Recommendations for '{movie}':")
        knn_recommendations = recommender.recommend(movie, method="knn")
        for title in knn_recommendations:
            print(title)
        print(f"\nNaive Bayes Recommendations for '{movie}':")
        nb_recommendations = recommender.recommend(movie, method="naive_bayes")
        for title in nb_recommendations:
            print(title)
    except FileNotFoundError:
        print("Error: The file 'top10K-TMDB-movies.csv' was not found.")
    except Exception as e:
        print(f"Error: {str(e)}")