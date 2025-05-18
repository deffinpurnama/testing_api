from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# === Load data ===
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'modified_dataframe.csv'))


# === Feature engineering ===
categorical_features = ['jenis', 'breed', 'gender', 'usia']
numeric_features = ['warna']

encoder = OneHotEncoder(sparse_output=False)
encoded_cat = encoder.fit_transform(df[categorical_features])
numeric_array = df[numeric_features].to_numpy()
feature_vectors = np.hstack([encoded_cat, numeric_array])

# === Mapping jenis -> list breed ===
breed_dict = df.groupby('jenis')['breed'].unique().apply(list).to_dict()

# === Fungsi rekomendasi ===
def recommend_by_preferences(preferences: dict, top_n=5):
    user_df = pd.DataFrame([preferences])
    encoded_user_cat = encoder.transform(user_df[categorical_features])
    user_num = np.array([[preferences['warna']]])
    user_vector = np.hstack([encoded_user_cat, user_num])

    similarity_scores = cosine_similarity(user_vector, feature_vectors)[0]
    top_indices = similarity_scores.argsort()[::-1][:top_n]

    rekomendasi = df.iloc[top_indices].copy()
    rekomendasi['similarity_score'] = similarity_scores[top_indices]
    return rekomendasi[['id', 'nama', 'jenis', 'breed', 'gender', 'usia', 'warna', 'similarity_score']]

# === Routing utama untuk HTML (lokal form) ===
@app.route('/', methods=['GET', 'POST'])
def index():
    rekomendasi = None
    if request.method == 'POST':
        user_input = {
            'jenis': request.form['jenis'],
            'breed': request.form['breed'],
            'gender': request.form['gender'],
            'usia': request.form['usia'],
            'warna': int(request.form['warna'])
        }
        rekomendasi = recommend_by_preferences(user_input, top_n=10).to_dict(orient='records')
    
    jenis_list = list(breed_dict.keys())
    return render_template('index.html', rekomendasi=rekomendasi, jenis_list=jenis_list)

# === Endpoint AJAX untuk ambil breed berdasarkan jenis ===
@app.route('/get_breeds')
def get_breeds():
    jenis = request.args.get('jenis')
    breeds = breed_dict.get(jenis, [])
    return jsonify(breeds)

# === Endpoint API untuk backend (POST JSON) ===
@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    data = request.get_json()

    # Validasi input
    required_fields = ['jenis', 'breed', 'gender', 'usia', 'warna']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing one or more required fields.'}), 400

    try:
        user_input = {
            'jenis': data['jenis'],
            'breed': data['breed'],
            'gender': data['gender'],
            'usia': data['usia'],
            'warna': int(data['warna'])
        }

        hasil_rekomendasi = recommend_by_preferences(user_input, top_n=10)
        return jsonify(hasil_rekomendasi.to_dict(orient='records'))

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === Run app ===
if __name__ == '__main__':
    app.run(debug=True)
