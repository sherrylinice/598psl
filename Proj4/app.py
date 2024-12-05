import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import load_npz

@st.cache_data
def load_movies():
    data_url = 'https://liangfgithub.github.io/MovieData/'
    movies = pd.read_csv(
        data_url + 'movies.dat?raw=true',
        sep='::',
        engine='python',
        names=['MovieID', 'Title', 'Genres'],
        encoding='latin-1'
    )
    movies['MovieID'] = movies['MovieID'].astype(int)
    movies.set_index('MovieID', inplace=True)
    return movies

@st.cache_data
def load_popular_movies():
    popular_movies = pd.read_csv('Proj4/popular_movies.csv')
    popular_movies['MovieID'] = popular_movies['MovieID'].astype(int)
    return popular_movies

@st.cache_resource
def load_resources():
    S_top = load_npz('Proj4/S_top.npz')
    with open('Proj4/movie_ids.pkl', 'rb') as f:
        movie_ids = pickle.load(f)
    with open('Proj4/movie_id_to_index.pkl', 'rb') as f:
        movie_id_to_index = pickle.load(f)
    return S_top, movie_ids, movie_id_to_index

def myIBCF(newuser):
    S_top, movie_ids, movie_id_to_index = load_resources()
    
    rated_indices = np.where(~np.isnan(newuser))[0]
    rated_ratings = newuser[rated_indices]

    predictions = {}
    for i in range(len(newuser)):
        if np.isnan(newuser[i]):
            start = S_top.indptr[i]
            end = S_top.indptr[i+1]
            indices = S_top.indices[start:end]
            data = S_top.data[start:end]
            if len(data) == 0:
                continue
            mask = np.in1d(indices, rated_indices)
            indices_filtered = indices[mask]
            data_filtered = data[mask]
            if len(data_filtered) == 0:
                continue
            ratings_filtered = newuser[indices_filtered]
            numerator = np.dot(data_filtered, ratings_filtered)
            denominator = np.sum(data_filtered)
            if denominator != 0:
                pred = numerator / denominator
                predictions[i] = pred

    if predictions:
        pred_series = pd.Series(predictions)
        pred_series = pred_series.sort_values(ascending=False)
        top_indices = pred_series.index[:10].tolist()
        top_predictions = pred_series.iloc[:10].tolist()
    else:
        top_indices = []
        top_predictions = []

    popular_movies = load_popular_movies()
    rated_movie_ids = [movie_ids[idx] for idx in rated_indices]
    additional_movies = popular_movies[~popular_movies['MovieID'].isin(rated_movie_ids)]
    additional_movie_ids = additional_movies['MovieID'].tolist()

    recommended_movie_ids = [movie_ids[idx] for idx in top_indices]
    predicted_ratings = top_predictions
    
    while len(recommended_movie_ids) < 10 and additional_movie_ids:
        next_movie_id = additional_movie_ids.pop(0)
        if next_movie_id not in recommended_movie_ids:
            recommended_movie_ids.append(next_movie_id)
            predicted_ratings.append(None)  

    top_ten_movie_names = ["m" + str(mid) for mid in recommended_movie_ids[:10]]
    predicted_ratings = predicted_ratings[:10]

    return top_ten_movie_names, predicted_ratings

# Load data
movies = load_movies()
popular_movies = load_popular_movies()
S_top, movie_ids, movie_id_to_index = load_resources()

# Base URL for poster images
poster_base_url = "https://liangfgithub.github.io/MovieImages/"

# App title
st.title('Movie Recommender')

# Custom CSS for the app style
st.markdown("""
    <style>
    .movie-title {
        min-height: 30px;
        text-align: center;
    }
    .stButton > button {
        background-color: #007bff;  /* Blue color */
        color: white;
    }
    .stButton > button:hover {
        background-color: #0056b3;  /* Darker blue on hover */
        color: white;
    }
    .stButton > button:active, .stButton > button:focus {
        background-color: #004080;  /* Even darker blue on click */
        color: white !important;  /* Force white text color */
    }
    h5 {
        font-size: 1.2rem;
        font-weight: bold;
    }
    div[data-baseweb="slider"] {
        margin-top: 5px;    
        margin-bottom: 0px;
        max-width: 90px;
        margin-left: auto;  
        margin-right: auto;
    }
    .element-container {
        margin-bottom: 0px;
    }
    .movie-poster {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .movie-row {
        border-top: 2px solid #ddd;
        padding-top: 0px;
        margin-top: 0px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h5>First, tell us what you love! Rate the following movies as many as possible.</h5>", unsafe_allow_html=True)
st.markdown("<div style='font-size: 1.2rem; color: #007bff; margin-bottom: 5px;'><strong>Next, scroll down to get your next favorite movie! 🎥</strong></div>", unsafe_allow_html=True)
st.markdown("<div style='font-size: 0.9rem; color: #6c757d;'>0 = Least Like, 5 = Most Like</div>", unsafe_allow_html=True)

newuser = np.full(len(movie_ids), np.nan)

user_ratings = {}
cols_per_row = 6  # Number of movies per row

# Limit the display to 100 movies
sample_movie_ids = popular_movies['MovieID'].head(100).tolist()

rows = len(sample_movie_ids) // cols_per_row + (len(sample_movie_ids) % cols_per_row > 0)
for row in range(rows):
    st.markdown('<div class="movie-row">', unsafe_allow_html=True)
    cols = st.columns(cols_per_row)
    for idx in range(cols_per_row):
        i = row * cols_per_row + idx
        if i >= len(sample_movie_ids):
            break
        mid = sample_movie_ids[i]
        if mid in movies.index:
            with cols[idx]:
                st.markdown('<div class="movie-poster">', unsafe_allow_html=True)
                title = movies.loc[mid, 'Title']
                poster_url = f"{poster_base_url}{mid}.jpg"
                st.image(poster_url, width=120)
                st.markdown(f"<div class='movie-title'><strong>{title}</strong></div>", unsafe_allow_html=True)
                rating = st.slider(
                    label="",
                    min_value=0,
                    max_value=5,
                    value=0,
                    step=1,
                    format="%d",
                    key=f"rating_{mid}",
                    label_visibility='collapsed'
                )
                st.markdown('</div>', unsafe_allow_html=True)
                if rating > 0:
                    user_ratings[mid] = rating
                    # Update newuser vector
                    idx_in_newuser = movie_id_to_index[mid]
                    newuser[idx_in_newuser] = rating
    st.markdown('</div>', unsafe_allow_html=True)

# Recommendation button
st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
if st.button('Get recommendations'):
    recommendations, ratings = myIBCF(newuser)
    st.write('Top 10 Movie Recommendations for You:')
    cols_per_row = 6
    for row in range(2):
        cols = st.columns(cols_per_row)
        for idx in range(cols_per_row):
            i = row * cols_per_row + idx
            if i >= len(recommendations):
                break
            mid = recommendations[i]
            if mid in movies.index:
                with cols[idx]:
                    title = movies.loc[mid, 'Title']
                    poster_url = f"{poster_base_url}{mid}.jpg"
                    st.image(poster_url, width=120)
                    st.write(f"**{title}**")
st.markdown('</div>', unsafe_allow_html=True)
