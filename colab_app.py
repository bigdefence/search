from flask import Flask, render_template, request, jsonify
from googleapiclient.discovery import build
import openai
import os
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
import faiss
from pyngrok import ngrok

# NLTK 데이터 다운로드
nltk.download('punkt')
nltk.download('stopwords')

# API 키 및 설정
OPENAI_API_KEY = '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'
CSE_ID = '033a330f222464c94'
GOOGLE_API_KEY = 'AIzaSyAhkSx_g17zPtPTbrjuGI6vwHAkiA_fqY4'
GEMINI_API = 'AIzaSyDcq3ZfAUo1i6_24CelEizJftuEkaAPz38'
NGROK_AUTH_TOKEN = '2XUv7KdIEJ5ivK2lA2ixgNS8P3Z_6c2rjjzgw6PpmnmX9C915'  # ngrok 인증 토큰

openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GEMINI_API)
gemini_model = genai.GenerativeModel('gemini-1.5-pro')

# BERT 모델 초기화
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

# Sentence Transformer 초기화
sentence_transformer = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

app = Flask(__name__)

# ngrok 설정
ngrok.set_auth_token('2iwL7znmJCg12tbH0HPJGLLdTzH_2KxTKLJ4qojehD7pNHjN')
public_url = ngrok.connect(5000)
print(f' * ngrok tunnel 개방 중: {public_url}')

def google_search(query, api_key, cse_id, num_results=20):
    service = build("customsearch", "v1", developerKey=api_key)
    result = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
    return result.get('items', [])

def fetch_content(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text[:5000]
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return ""

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    return ' '.join([w for w in word_tokens if w.isalnum() and w not in stop_words])

def semantic_search(query, documents, top_k=5):
    query_embedding = sentence_transformer.encode([query])
    document_embeddings = sentence_transformer.encode(documents)
    dimension = query_embedding.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(document_embeddings)
    distances, indices = index.search(query_embedding, top_k)
    return indices[0]

def chatgpt_query(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an advanced AI assistant that provides comprehensive and insightful answers based on search results. Your task is to analyze the given information, synthesize it, and present a well-structured response in Korean. Include relevant facts, explain their significance, and provide any additional context or insights that would be valuable to the user. If there are conflicting viewpoints, present them objectively. Your response should be informative, engaging, and tailored to the user's query. Always cite your sources using [1], [2], etc., corresponding to the search result numbers."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=2000,
    )
    return response.choices[0].message.content.strip()

def gemini_query(prompt):
    system_prompt = """
    You are an advanced AI assistant tasked with providing comprehensive and insightful answers based on search results. Analyze the given information, synthesize it, and present a well-structured response in Korean. Your response should:

    1. Summarize the key points from the search results
    2. Explain the significance of the information
    3. Provide additional context or background information
    4. Highlight any conflicting viewpoints, if present
    5. Offer insights or implications related to the topic
    6. Suggest areas for further exploration, if applicable
    7. Always cite your sources using [1], [2], etc., corresponding to the search result numbers

    Your response should be informative, engaging, and directly relevant to the user's query. Aim to provide a response that goes beyond simple summarization, offering a deeper understanding of the topic.
    """
    full_prompt = f"{system_prompt}\n\nUser query and search results:\n{prompt}"
    response = gemini_model.generate_content(full_prompt)
    return response.text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        model_choice = request.form['model']
        
        search_results = google_search(query, GOOGLE_API_KEY, CSE_ID, num_results=20)
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_url = {executor.submit(fetch_content, result['link']): result for result in search_results}
            for future in as_completed(future_to_url):
                result = future_to_url[future]
                try:
                    content = future.result()
                    result['content'] = content
                    result['preprocessed_content'] = preprocess_text(content)
                except Exception as e:
                    print(f"Error processing content: {e}")
                    result['content'] = ""
                    result['preprocessed_content'] = ""

        documents = [result['preprocessed_content'] for result in search_results]
        top_indices = semantic_search(query, documents)

        prompt = f"User query: {query}\n\nSearch results:\n"
        for i, index in enumerate(top_indices, 1):
            result = search_results[index]
            title = result.get('title', 'No title')
            snippet = result.get('snippet', 'No description available.')
            link = result.get('link', 'No link available')
            content = result.get('content', 'No content available')
            prompt += f"{i}. Title: {title}\nDescription: {snippet}\nLink: {link}\nContent: {content[:1000]}\n\n"
        
        prompt += "Based on these search results and content, provide a comprehensive answer to the user's query. Your response should be in Korean, and include relevant information, explanations, and insights. Always cite your sources using [1], [2], etc., corresponding to the search result numbers."

        if model_choice == 'ChatGPT':
            response = chatgpt_query(prompt)
        elif model_choice == 'Gemini':
            response = gemini_query(prompt)
        
        return render_template('index.html', query=query, response=response, search_results=search_results)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=5000)