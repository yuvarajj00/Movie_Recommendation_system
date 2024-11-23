from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

app = Flask(__name__)

# Load and preprocess data
movies_data = pd.read_csv('movies.csv')
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

vectorizer = TfidfVectorizer()
feature_vector = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vector)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form['movie_name']
    list_of_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_titles, n=1)
    
    if not find_close_match:
        return render_template('recommendations.html', movies=[], error="Movie not found. Please try again.")
    
    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    
    recommended_movies = []
    for movie in sorted_similar_movies[1:31]:  # Skip the first movie (input movie)
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        recommended_movies.append(title_from_index)
    
    return render_template('recommendations.html', movies=recommended_movies, error=None)

if __name__ == '__main__':
    app.run(debug=True)
