import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# Load and preprocess data
data = pd.read_csv("top10K-TMDB-movies.csv")
data = data.fillna('')
data['soup'] = data['title'] + ' ' + data['genre']

# Vectorize the 'soup' column
vectorizer = CountVectorizer(max_features=5000, stop_words='english', lowercase=True)
X = vectorizer.fit_transform(data['soup'])
print(f"Feature matrix shape: {X.shape}")

# Optimized Cosine Similarity computation using sklearn
# This is much more memory efficient than converting to dense
cos_sim_matrix = cosine_similarity(X)
print(f"Cosine similarity matrix computed with shape: {cos_sim_matrix.shape}")

# Alternative: Custom sparse cosine similarity (if you prefer custom implementation)
def compute_cosine_similarity_sparse(X):
    """
    Compute cosine similarity for sparse matrix without converting to dense.
    More memory efficient for large datasets.
    """
    # Normalize the sparse matrix
    X_normalized = X.copy().astype(np.float64)
    # Compute norms for each row
    norms = np.sqrt(np.array(X_normalized.multiply(X_normalized).sum(axis=1)).flatten())
    # Avoid division by zero
    norms[norms == 0] = 1
    # Normalize each row
    X_normalized = X_normalized.multiply(1 / norms[:, np.newaxis])
    # Compute cosine similarity
    cos_sim = X_normalized @ X_normalized.T
    return cos_sim.toarray()

# Use the custom implementation if preferred
# cos_sim_matrix = compute_cosine_similarity_sparse(X)

# Optimized Nearest Neighbors implementation
def find_nearest_neighbors(movie_idx, similarity_matrix, n_neighbors=6):
    """
    Find the n_neighbors most similar movies based on cosine similarity.
    Returns indices and similarity scores.
    """
    sim_scores = similarity_matrix[movie_idx]
    # Get indices sorted by similarity (descending), exclude self (index 0)
    indices = np.argsort(sim_scores)[::-1][1:n_neighbors]
    similarities = sim_scores[indices]
    distances = 1 - similarities  # Convert similarity to distance
    return distances, indices

# Custom Multinomial Naive Bayes implementation
class CustomMultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.class_priors = None
        self.feature_probs = None

    def fit(self, X, y):
        # Convert X to dense if sparse (only for small matrices)
        if hasattr(X, 'toarray') and X.shape[0] < 1000:
            X_dense = X.toarray()
        else:
            X_dense = X
        
        self.classes = np.unique(y)
        n_samples, n_features = X_dense.shape
        n_classes = len(self.classes)
        
        # Initialize counts
        self.class_counts = np.zeros(n_classes)
        self.feature_counts = np.zeros((n_classes, n_features))
        
        # Compute counts
        for i, c in enumerate(self.classes):
            mask = (y == c).values if hasattr(y, 'values') else (y == c)
            mask_indices = np.where(mask)[0]
            
            if hasattr(X, 'toarray'):
                X_c = X[mask_indices].toarray()
            else:
                X_c = X_dense[mask_indices]
            
            self.class_counts[i] = X_c.shape[0]
            self.feature_counts[i, :] = X_c.sum(axis=0)
        
        # Compute priors and likelihoods with smoothing
        self.class_priors = self.class_counts / n_samples
        self.feature_probs = (self.feature_counts + self.alpha) / (
            self.feature_counts.sum(axis=1)[:, np.newaxis] + self.alpha * n_features
        )
    
    def predict(self, X):
        if hasattr(X, 'toarray') and X.shape[0] < 1000:
            X_dense = X.toarray()
        else:
            X_dense = X
            
        log_priors = np.log(self.class_priors)
        
        if hasattr(X, 'toarray'):
            log_likelihoods = X @ np.log(self.feature_probs.T)
        else:
            log_likelihoods = X_dense @ np.log(self.feature_probs.T)
            
        log_posteriors = log_priors + log_likelihoods
        return self.classes[np.argmax(log_posteriors, axis=1)]

# Train Naive Bayes (for demonstration)
y = data['genre'].apply(lambda x: x.split('|')[0] if '|' in x else x)
nb = CustomMultinomialNB()
nb.fit(X, y)
print(f"Naive Bayes trained on {len(np.unique(y))} genre classes")

# Updated recommend function using optimized implementations
def recommend(title, top_n=5):
    """
    Recommend movies based on title similarity using multiple approaches.
    """
    # Find movie index
    movie_matches = data[data['title'].str.lower() == title.lower()]
    if movie_matches.empty:
        print(f"Movie '{title}' not found in dataset.")
        # Try partial matching
        partial_matches = data[data['title'].str.lower().str.contains(title.lower(), na=False)]
        if not partial_matches.empty:
            print(f"Did you mean one of these? {partial_matches['title'].tolist()}")
        return []
    
    idx = movie_matches.index[0]
    print(f"Found movie: {data.iloc[idx]['title']} (Index: {idx})")
    
    # Cosine Similarity recommendations
    sim_scores = list(enumerate(cos_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    cos_recommendations = [(data.iloc[i]['title'], score) for i, score in sim_scores[1:top_n+1]]
    
    # Custom KNN recommendations
    distances, indices = find_nearest_neighbors(idx, cos_sim_matrix, n_neighbors=top_n+1)
    knn_recommendations = [(data.iloc[i]['title'], 1-distances[j]) for j, i in enumerate(indices)]
    
    # Combine and rank recommendations
    all_recommendations = {}
    
    # Add cosine similarity recommendations
    for title_rec, score in cos_recommendations:
        if title_rec not in all_recommendations:
            all_recommendations[title_rec] = []
        all_recommendations[title_rec].append(score)
    
    # Add KNN recommendations
    for title_rec, score in knn_recommendations:
        if title_rec not in all_recommendations:
            all_recommendations[title_rec] = []
        all_recommendations[title_rec].append(score)
    
    # Calculate final scores (average of different methods)
    final_recommendations = []
    for title_rec, scores in all_recommendations.items():
        avg_score = np.mean(scores)
        final_recommendations.append((title_rec, avg_score))
    
    # Sort by score and return top N
    final_recommendations.sort(key=lambda x: x[1], reverse=True)
    
    return [[title,score] for title, score in final_recommendations[:top_n]]

# Example usage
print("\n" + "="*50)
print("MOVIE RECOMMENDATION SYSTEM")
print("="*50)

test_movie = "Star wars"
print(f"\nGetting recommendations for: '{test_movie}'")
recommendations = recommend(test_movie, top_n=5)

if recommendations:
    print(f"\nTop 5 recommended movies similar to '{test_movie}':")
    for i in range(len(recommendations)):
        print(f"{i}. {recommendations[i][0]} - {recommendations[i][1]}")
else:
    print("No recommendations found.")

# Display some statistics
print(f"\nDataset Statistics:")
print(f"Total movies: {len(data)}")
print(f"Unique genres: {len(data['genre'].str.split('|').explode().unique())}")
print(f"Feature matrix sparsity: {(1 - X.nnz / (X.shape[0] * X.shape[1])) * 100:.2f}%")