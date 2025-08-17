import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from textblob import TextBlob
from transformers import pipeline
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import re
import demoji
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_function' not in st.session_state:
    st.session_state.current_function = "general"

# Load models and data
@st.cache_resource
def load_models():
    """Load all pre-trained models and data"""
    models = {}
    
    # Load Book Recommender
    try:
        models['popular_books'] = pickle.load(open('popular.pkl', 'rb'))
        models['pt'] = pickle.load(open('pt.pkl', 'rb'))
        models['books'] = pickle.load(open('books.pkl', 'rb'))
        models['similarity_scores'] = pickle.load(open('similarity_scores.pkl', 'rb'))
    except:
        st.warning("Book recommender models not found")
    
    # Load Movie Recommender
    try:
        models['movies_dict'] = pickle.load(open('movies_dict.pkl', 'rb'))
        models['movie_similarity'] = pickle.load(open('similarity.pkl', 'rb'))
    except:
        st.warning("Movie recommender models not found")
    
    # Load Fashion Model
    try:
        base_model = VGG16(weights='imagenet', include_top=False)
        models['fashion_model'] = Model(inputs=base_model.input, outputs=base_model.output)
    except:
        st.warning("Fashion model not loaded")
    
    # Load IPL Model
    try:
        models['ipl_model'] = pickle.load(open('pipe.pkl', 'rb'))
    except:
        st.warning("IPL prediction model not found")
    
    return models

# Initialize Spotify client
@st.cache_resource
def init_spotify():
    try:
        client_credentials_manager = SpotifyClientCredentials(
            client_id="your_client_id",
            client_secret="your_client_secret"
        )
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        return sp
    except:
        return None

# Helper functions
def preprocess_image(img_path):
    """Preprocess image for fashion recommendation"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def extract_features(model, preprocessed_img):
    """Extract features from image"""
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features

def get_sentiment(text):
    """Analyze sentiment of text"""
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Chatbot functions
class MultiFunctionChatbot:
    def __init__(self):
        self.models = load_models()
        self.spotify = init_spotify()
        
    def general_chat(self, message):
        """General conversation responses"""
        greetings = ["hello", "hi", "hey", "greetings"]
        questions = ["how are you", "what can you do", "help", "capabilities"]
        
        message_lower = message.lower()
        
        if any(greet in message_lower for greet in greetings):
            return "Hello! I'm your multi-function AI assistant. I can help with book recommendations, movie suggestions, fashion advice, music recommendations, IPL predictions, and more!"
        elif any(q in message_lower for q in questions):
            return "I can help you with:\n📚 Book recommendations\n🎬 Movie suggestions\n👗 Fashion recommendations\n🎵 Music discovery\n🏏 IPL match predictions\n💬 WhatsApp chat analysis\n📺 YouTube chaptering\nJust tell me what you'd like to explore!"
        else:
            return "I'm here to help! Try asking me about books, movies, fashion, music, or sports predictions."
    
    def book_recommendation(self, book_name=None):
        """Book recommendation system"""
        if 'pt' not in self.models or 'similarity_scores' not in self.models:
            return "Book recommender not available. Please ensure models are loaded."
        
        try:
            if book_name:
                # Get similar books
                pt = self.models['pt']
                similarity_scores = self.models['similarity_scores']
                books = self.models['books']
                
                if book_name in pt.index:
                    index = np.where(pt.index == book_name)[0][0]
                    similar_items = sorted(list(enumerate(similarity_scores[index])), 
                                         key=lambda x: x[1], reverse=True)[1:6]
                    
                    recommendations = []
                    for i in similar_items:
                        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
                        recommendations.append({
                            'title': temp_df['Book-Title'].values[0],
                            'author': temp_df['Book-Author'].values[0],
                            'image': temp_df['Image-URL-M'].values[0]
                        })
                    return recommendations
                else:
                    return f"Book '{book_name}' not found in database."
            else:
                # Return popular books
                popular_df = self.models['popular_books']
                return popular_df[['Book-Title', 'Book-Author', 'Image-URL-M']].head(10).to_dict('records')
        except Exception as e:
            return f"Error in book recommendation: {str(e)}"
    
    def movie_recommendation(self, movie_name):
        """Movie recommendation system"""
        if 'movies_dict' not in self.models:
            return "Movie recommender not available."
        
        try:
            movies_dict = self.models['movies_dict']
            similarity = self.models['movie_similarity']
            
            movies_df = pd.DataFrame(movies_dict)
            
            if movie_name in movies_df['title'].values:
                movie_index = movies_df[movies_df['title'] == movie_name].index[0]
                distances = similarity[movie_index]
                movies_list = sorted(list(enumerate(distances)), reverse=True, 
                                   key=lambda x: x[1])[1:6]
                
                recommendations = []
                for i in movies_list:
                    recommendations.append(movies_df.iloc[i[0]]['title'])
                return recommendations
            else:
                return f"Movie '{movie_name}' not found in database."
        except Exception as e:
            return f"Error in movie recommendation: {str(e)}"
    
    def fashion_recommendation(self, uploaded_image):
        """Fashion recommendation system"""
        if 'fashion_model' not in self.models:
            return "Fashion recommender not available."
        
        try:
            # This would need pre-computed features for all fashion items
            # For demo purposes, return sample recommendations
            return ["Fashion recommendation requires pre-computed image database"]
        except Exception as e:
            return f"Error in fashion recommendation: {str(e)}"
    
    def music_recommendation(self, song_name):
        """Music recommendation system"""
        if not self.spotify:
            return "Spotify integration not available. Please configure credentials."
        
        try:
            # Search for the song
            results = self.spotify.search(q=song_name, type='track', limit=1)
            if results['tracks']['items']:
                track = results['tracks']['items'][0]
                track_id = track['id']
                
                # Get recommendations
                recommendations = self.spotify.recommendations(seed_tracks=[track_id], limit=5)
                
                songs = []
                for track in recommendations['tracks']:
                    songs.append({
                        'name': track['name'],
                        'artist': track['artists'][0]['name'],
                        'album': track['album']['name'],
                        'preview_url': track['preview_url']
                    })
                return songs
            else:
                return f"Song '{song_name}' not found on Spotify."
        except Exception as e:
            return f"Error in music recommendation: {str(e)}"
    
    def ipl_prediction(self, batting_team, bowling_team, runs_left, balls_left, wickets, total_runs, crr, rrr):
        """IPL match prediction"""
        if 'ipl_model' not in self.models:
            return "IPL prediction model not available."
        
        try:
            input_data = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'runs_left': [runs_left],
                'balls_left': [balls_left],
                'wickets': [wickets],
                'total_runs': [total_runs],
                'crr': [crr],
                'rrr': [rrr]
            })
            
            prediction = self.models['ipl_model'].predict_proba(input_data)[0]
            win_prob = prediction[1] * 100
            lose_prob = prediction[0] * 100
            
            return {
                'win_probability': round(win_prob, 2),
                'lose_probability': round(lose_prob, 2)
            }
        except Exception as e:
            return f"Error in IPL prediction: {str(e)}"
    
    def whatsapp_analysis(self, uploaded_file):
        """WhatsApp chat analysis"""
        try:
            content = uploaded_file.read().decode('utf-8')
            
            # Basic analysis
            messages = content.split('\n')
            total_messages = len([m for m in messages if m.strip()])
            
            # Sentiment analysis
            sentiments = []
            for message in messages[:100]:  # Limit for performance
                if ':' in message:
                    text = message.split(':', 1)[1]
                    sentiments.append(get_sentiment(text))
            
            sentiment_counts = pd.Series(sentiments).value_counts().to_dict()
            
            return {
                'total_messages': total_messages,
                'sentiment_distribution': sentiment_counts,
                'sample_messages': messages[:5]
            }
        except Exception as e:
            return f"Error in WhatsApp analysis: {str(e)}"

# Initialize chatbot
chatbot = MultiFunctionChatbot()

# Streamlit UI
st.set_page_config(page_title="Multi-Function AI Assistant", page_icon="🤖", layout="wide")

st.title("🤖 Multi-Function AI Assistant")
st.markdown("Your one-stop solution for recommendations, predictions, and analysis!")

# Sidebar for function selection
with st.sidebar:
    st.header("Select Function")
    function = st.selectbox(
        "Choose a feature:",
        ["General Chat", "Book Recommendations", "Movie Recommendations", 
         "Fashion Recommendations", "Music Recommendations", "IPL Predictions", 
         "WhatsApp Analysis"]
    )

# Main interface
if function == "General Chat":
    st.header("💬 General Chat")
    
    user_input = st.text_input("Ask me anything:")
    if st.button("Send"):
        if user_input:
            response = chatbot.general_chat(user_input)
            st.write("**Assistant:**", response)

elif function == "Book Recommendations":
    st.header("📚 Book Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Popular Books")
        popular_books = chatbot.book_recommendation()
        if isinstance(popular_books, list):
            for book in popular_books[:5]:
                st.write(f"**{book['title']}** by {book['author']}")
    
    with col2:
        st.subheader("Search for Similar Books")
        book_name = st.text_input("Enter a book title:")
        if st.button("Get Recommendations"):
            if book_name:
                recommendations = chatbot.book_recommendation(book_name)
                if isinstance(recommendations, list):
                    for book in recommendations:
                        st.write(f"**{book['title']}** by {book['author']}")
                else:
                    st.write(recommendations)

elif function == "Movie Recommendations":
    st.header("🎬 Movie Recommendations")
    
    movie_name = st.text_input("Enter a movie title:")
    if st.button("Get Recommendations"):
        if movie_name:
            recommendations = chatbot.movie_recommendation(movie_name)
            if isinstance(recommendations, list):
                st.write("**Recommended Movies:**")
                for movie in recommendations:
                    st.write(f"- {movie}")
            else:
                st.write(recommendations)

elif function == "Fashion Recommendations":
    st.header("👗 Fashion Recommendations")
    
    uploaded_file = st.file_uploader("Upload an image:", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Get Fashion Recommendations"):
            recommendations = chatbot.fashion_recommendation(uploaded_file)
            st.write(recommendations)

elif function == "Music Recommendations":
    st.header("🎵 Music Recommendations")
    
    song_name = st.text_input("Enter a song name:")
    if st.button("Get Recommendations"):
        if song_name:
            recommendations = chatbot.music_recommendation(song_name)
            if isinstance(recommendations, list):
                for song in recommendations:
                    st.write(f"**{song['name']}** by {song['artist']}")
                    if song['preview_url']:
                        st.audio(song['preview_url'])
            else:
                st.write(recommendations)

elif function == "IPL Predictions":
    st.header("🏏 IPL Match Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        batting_team = st.selectbox("Batting Team:", 
                                  ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
                                   'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
                                   'Rajasthan Royals', 'Delhi Capitals'])
        runs_left = st.number_input("Runs Left:", min_value=0, max_value=300, value=50)
        wickets = st.number_input("Wickets Left:", min_value=0, max_value=10, value=5)
        crr = st.number_input("Current Run Rate:", min_value=0.0, max_value=20.0, value=6.0)
    
    with col2:
        bowling_team = st.selectbox("Bowling Team:", 
                                  ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
                                   'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
                                   'Rajasthan Royals', 'Delhi Capitals'])
        balls_left = st.number_input("Balls Left:", min_value=0, max_value=120, value=30)
        total_runs = st.number_input("Target Runs:", min_value=0, max_value=300, value=150)
        rrr = st.number_input("Required Run Rate:", min_value=0.0, max_value=20.0, value=10.0)
    
    if st.button("Predict Match Outcome"):
        prediction = chatbot.ipl_prediction(batting_team, bowling_team, runs_left, 
                                          balls_left, wickets, total_runs, crr, rrr)
        if isinstance(prediction, dict):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Win Probability", f"{prediction['win_probability']}%")
            with col2:
                st.metric("Lose Probability", f"{prediction['lose_probability']}%")

elif function == "WhatsApp Analysis":
    st.header("💬 WhatsApp Chat Analysis")
    
    uploaded_file = st.file_uploader("Upload WhatsApp chat file (.txt):", type=['txt'])
    if uploaded_file is not None:
        analysis = chatbot.whatsapp_analysis(uploaded_file)
        
        if isinstance(analysis, dict):
            st.write(f"**Total Messages:** {analysis['total_messages']}")
            
            st.subheader("Sentiment Distribution")
            sentiment_df = pd.DataFrame(list(analysis['sentiment_distribution'].items()), 
                                      columns=['Sentiment', 'Count'])
            st.bar_chart(sentiment_df.set_index('Sentiment'))
            
            st.subheader("Sample Messages")
            for msg in analysis['sample_messages']:
                st.text(msg)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | All portfolio functionalities integrated into one powerful assistant")
