from flask import Flask, render_template, request, redirect, url_for, session
from pymongo import MongoClient
import os
import spacy
from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModel
import torch

app = Flask(__name__)
app.secret_key = os.urandom(24)

logging.basicConfig(level=logging.DEBUG)


logging.debug("MongoDB bağlantısı kuruluyor...")
client = MongoClient("mongodb+srv://grknclf1907:12345@gcveri.2nmlzts.mongodb.net/")
db = client["UserData"]
collection = db["User"]

logging.debug("spaCy modeli yükleniyor...")
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

logging.debug("FastText modeli yükleniyor...")
fasttext_model = FastText.load_fasttext_format("C:\\Users\\grknc\\Desktop\\cc.en.300.bin")

logging.debug("SciBERT modeli yükleniyor...")
scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
scibert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")


def load_documents():
    documents = []
    folder_path = 'C:/Users/grknc/Desktop/grknclf/okul/Krapivin2009/Krapivin2009/docsutf8'
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                article = parse_document(content, filename)
                if article:
                    documents.append(article)
    return documents

def parse_document(content, filename):

    title_start = content.find('--T') + 3
    title_end = content.find('--A')
    title = content[title_start:title_end].strip()

    abstract_start = content.find('--A') + 3
    abstract_end = content.find('--B')
    abstract = content[abstract_start:abstract_end].strip()

    file_id = filename.replace('.txt', '')

    if title and abstract:
        return {
            'id': file_id,
            'title': title,
            'abstract': abstract
        }
    else:
        return None

articles = load_documents()

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def preprocess_texts_parallel(texts, num_workers=4):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        cleaned_texts = list(executor.map(preprocess_text, texts))
    return cleaned_texts

documents = [article['abstract'] for article in articles]
cleaned_documents = preprocess_texts_parallel(documents)

def get_fasttext_vector(text):
    words = text.split()
    vectors = [fasttext_model.wv[word] for word in words if word in fasttext_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(fasttext_model.vector_size)

fasttext_vectors = [get_fasttext_vector(doc) for doc in cleaned_documents]

def get_scibert_vector(text):
    inputs = scibert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = scibert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]

scibert_vectors = [get_scibert_vector(doc) for doc in cleaned_documents]

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = collection.find_one({'username': request.form['username'], 'password': request.form['password']})
        if user:
            session['username'] = user['username']
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Login failed. Please check your username and password.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

@app.route('/index')
def index():
    if 'username' in session:
        username = session['username']
        return render_template('index.html', username=username)
    else:
        return redirect(url_for('home'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        interests = request.form['interests'].split(',')
        new_user = {
            'first_name': request.form['first_name'],
            'last_name': request.form['last_name'],
            'username': request.form['username'],
            'password': request.form['password'],
            'interests': interests
        }
        collection.insert_one(new_user)
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'username' not in session:
        return redirect(url_for('home'))

    username = session['username']
    if request.method == 'POST':
        collection.update_one({'username': username}, {'$set': {
            'first_name': request.form['first_name'],
            'last_name': request.form['last_name'],
            'interests': request.form['interests'].split(',')
        }})
        return redirect(url_for('profile'))

    user = collection.find_one({'username': username})
    return render_template('profile.html', user=user)

def calculate_precision(recommended_docs, selected_docs):
    true_positives = len(set(recommended_docs) & set(selected_docs))
    precision = true_positives / len(recommended_docs) if recommended_docs else 0
    return precision


@app.route('/academic_paper_system', methods=['GET', 'POST'])
def academic_paper_system():
    logging.debug("Academic Paper System sayfası açılıyor...")
    if 'username' not in session:
        return redirect(url_for('home'))

    user = collection.find_one({'username': session['username']})
    user_interests = " ".join(user['interests'])

    selected_vector = None
    selected_scibert_vector = None
    selected_fasttext_doc_ids = []
    selected_scibert_doc_ids = []
    selected_fasttext_docs = []
    selected_scibert_docs = []

    if request.method == 'POST':
        selected_fasttext_doc_ids = request.form.getlist('selected_fasttext_docs')
        selected_scibert_doc_ids = request.form.getlist('selected_scibert_docs')
        selected_fasttext_docs = [article['abstract'] for article in articles if article['id'] in selected_fasttext_doc_ids]
        selected_scibert_docs = [article['abstract'] for article in articles if article['id'] in selected_scibert_doc_ids]
        if selected_fasttext_docs:
            selected_vector = np.mean([get_fasttext_vector(preprocess_text(doc)) for doc in selected_fasttext_docs], axis=0)
        else:
            selected_vector = get_fasttext_vector(preprocess_text(user_interests))

        if selected_scibert_docs:
            selected_scibert_vector = np.mean([get_scibert_vector(preprocess_text(doc)) for doc in selected_scibert_docs], axis=0)
        else:
            selected_scibert_vector = get_scibert_vector(preprocess_text(user_interests))
    else:
        selected_vector = get_fasttext_vector(preprocess_text(user_interests))
        selected_scibert_vector = get_scibert_vector(preprocess_text(user_interests))

    fasttext_similarities = cosine_similarity([selected_vector], fasttext_vectors)[0]
    scibert_similarities = cosine_similarity([selected_scibert_vector], scibert_vectors)[0]

    top_5_fasttext_indices = fasttext_similarities.argsort()[-5:][::-1]
    top_5_scibert_indices = scibert_similarities.argsort()[-5:][::-1]

    top_5_fasttext_documents = [(articles[i], fasttext_similarities[i]) for i in top_5_fasttext_indices]
    top_5_scibert_documents = [(articles[i], scibert_similarities[i]) for i in top_5_scibert_indices]

    # Precision hesaplaması
    fasttext_recommended_ids = [articles[i]['id'] for i in top_5_fasttext_indices]
    scibert_recommended_ids = [articles[i]['id'] for i in top_5_scibert_indices]

    fasttext_precision = calculate_precision(fasttext_recommended_ids, selected_fasttext_doc_ids)
    scibert_precision = calculate_precision(scibert_recommended_ids, selected_scibert_doc_ids)

    logging.debug("Top 5 makale önerisi oluşturuldu.")
    return render_template('academic_paper_system.html',
                           fasttext_docs=top_5_fasttext_documents,
                           scibert_docs=top_5_scibert_documents,
                           fasttext_precision=fasttext_precision,
                           scibert_precision=scibert_precision)

@app.route('/search', methods=['GET', 'POST'])
def search():
    query = request.form.get('query')
    results = []
    if query:
        results = [article for article in articles if query.lower() in article['title'].lower()]
    return render_template('search.html', query=query, results=results)

if __name__ == '__main__':
    app.run(debug=True)
