from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
import re
from functools import lru_cache
import random

app = Flask(__name__)

# Fungsi untuk membaca kata kunci dari file teks
def read_keywords(file_path):
    with open(file_path, 'r') as file:
        keywords = [line.strip() for line in file.readlines()]
    return keywords

# Memanggil fungsi untuk membaca kata kunci
keywords = read_keywords('keywords.txt')  # Ganti dengan path yang benar ke file keywords.txt

# Initialize Firestore database
cred = credentials.Certificate('sportspot-d0d68-firebase-adminsdk-i7wnv-d8563c7c4e.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load model
model = load_model('model.h5')

# Contoh mapping tema ke indeks
label_mapping = {'badminton': 0, 'basket': 1, 'voli': 2}

# Inisialisasi dan konfigurasi tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(keywords)

# Fungsi untuk membersihkan judul artikel (contoh sederhana)
def clean_title(title):
    title = title.lower()  # Convert to lowercase
    title = re.sub(r'\d+', '', title)  # Remove numbers
    title = re.sub(r'[^\w\s]', '', title)  # Remove non-alphanumeric characters
    return title

# Fungsi untuk melakukan prediksi tema berdasarkan judul artikel
def predict_theme(title):
    cleaned_title = clean_title(title)
    tokenized_title = tokenizer.texts_to_sequences([cleaned_title])
    padded_tokenized_title = pad_sequences(tokenized_title, maxlen=10)  # Sesuaikan maxlen sesuai dengan model Anda
    prediction = model.predict(padded_tokenized_title)[0]
    predicted_theme_index = np.argmax(prediction)
    predicted_theme = list(label_mapping.keys())[predicted_theme_index]
    return predicted_theme

# Cache untuk menyimpan hasil prediksi
@lru_cache(maxsize=128)
def get_predicted_theme(title):
    return predict_theme(title)

# Fungsi untuk mendapatkan artikel secara acak
def get_random_articles(articles, num_articles=5):
    article_list = list(articles.values())
    random_articles = random.sample(article_list, min(num_articles, len(article_list)))
    return [{
        'title': article.get('title', ''),
        'link': article.get('link', ''),
        'imageLink': article.get('imageLink', ''),
        'time': article.get('time', '')
    } for article in random_articles]

# Fungsi untuk mendapatkan artikel berdasarkan tema acak
def get_articles_by_random_theme(articles, num_articles=5):
    random_theme = random.choice(list(label_mapping.keys()))
    theme_articles = []
    for article_id, article_data in articles.items():
        if get_predicted_theme(article_data.get('title', '')) == random_theme:
            theme_articles.append({
                'title': article_data.get('title', ''),
                'link': article_data.get('link', ''),
                'imageLink': article_data.get('imageLink', ''),
                'time': article_data.get('time', '')
            })
        if len(theme_articles) >= num_articles:
            break
    return theme_articles

# Endpoint untuk merekomendasikan artikel berdasarkan riwayat pengguna
@app.route('/recommend_articles', methods=['POST'])
def recommend_articles():
    user_id = request.json.get('userId')

    if not user_id:
        return jsonify({'error': 'Missing userId in request'}), 400

    try:
        # Mendapatkan riwayat artikel pengguna dari Firestore berdasarkan userId
        user_history_ref = db.collection('userHistory').where('userId', '==', user_id).stream()
        user_articles = [doc.to_dict().get('articleId') for doc in user_history_ref]

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Mendapatkan semua artikel dari koleksi articles di Firestore
    try:
        articles_ref = db.collection('articles').stream()
        articles = {article.id: article.to_dict() for article in articles_ref}

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Jika tidak ada riwayat artikel pengguna, kembalikan artikel berdasarkan tema acak
    if not user_articles:
        random_articles = get_random_articles(articles)
        return jsonify({'recommended_articles': random_articles})

    recommended_articles = []

    # Mengiterasi melalui artikel-artikel yang dilihat pengguna
    for article_id in user_articles:
        article = articles.get(article_id)
        if article:
            title = article.get('title', '')
            predicted_theme = get_predicted_theme(title)
            # Mengumpulkan artikel yang sesuai dengan tema yang diprediksi
            for article_id, article_data in articles.items():
                if get_predicted_theme(article_data.get('title', '')) == predicted_theme:
                    recommended_articles.append({
                        'title': article_data.get('title', ''),
                        'link': article_data.get('link', ''),
                        'imageLink': article_data.get('imageLink', ''),
                        'time': article_data.get('time', '')
                    })
                if len(recommended_articles) >= 5:
                    break
        if len(recommended_articles) >= 5:
            break

    return jsonify({'recommended_articles': recommended_articles[:5]})

# Endpoint untuk mengambil semua data dari koleksi userHistory
@app.route('/user_history', methods=['GET'])
def get_user_history():
    try:
        user_history_ref = db.collection('userHistory').get()
        
        user_history_data = []
        for doc in user_history_ref:
            user_history_data.append({
                'id': doc.id,
                'articleId': doc.get('articleId'),
                'userId': doc.get('userId')
            })

        return jsonify({'user_history': user_history_data}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

