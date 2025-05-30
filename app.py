from flask import Flask, request, jsonify, render_template
from movie_recommender import MovieRecommender

app = Flask(__name__)
recommender = MovieRecommender("top10K-TMDB-movies.csv")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend")
def recommend():
    title = request.args.get("title", "")
    method = request.args.get("method", "knn")
    try:
        results = recommender.recommend(title, method=method)
        return jsonify(results)
    except Exception as e:
        return jsonify([])

@app.route("/autocomplete")
def autocomplete():
    query = request.args.get("q", "").lower()
    if len(query) <= 2:
        return jsonify([])

    matches = [
        movie['title'] for movie in recommender.movies
        if isinstance(movie['title'], str) and movie['title'].lower().startswith(query)
    ][:10] 
    return jsonify(matches)

if __name__ == "__main__":
    app.run(debug=True)