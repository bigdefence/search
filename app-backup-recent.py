from flask import Flask, render_template, request, jsonify
from googleapiclient.discovery import build
from dotenv import load_dotenv
import openai
import os
import google.generativeai as genai
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from stopwords import get_stopwords
from konlpy.tag import Okt
from sentence_transformers import SentenceTransformer
import numpy as np
import sqlite3
from functools import lru_cache
import faiss
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()
okt = Okt()

# API keys and environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CSE_ID = os.getenv('CSE_ID2')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
NAVER_CLIENT_ID = os.getenv('NAVER_CLIENT_ID')
NAVER_CLIENT_SECRET = os.getenv('NAVER_CLIENT_SECRET')

# OpenAI and Gemini setup
openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Korean stopwords setup
korean_stopwords = set(get_stopwords('ko'))

# Sentence transformer model for semantic search
sentence_transformer = SentenceTransformer('distiluse-base-multilingual-cased-v2')
d = 512  # dimensionality of sentence embeddings
quantizer = faiss.IndexFlatL2(d)
faiss_index = faiss.IndexIDMap(quantizer)

# TF-IDF Vectorizer for keyword extraction
tfidf_vectorizer = TfidfVectorizer(stop_words=list(korean_stopwords))

app = Flask(__name__)

# Database initialization
def init_db():
    with sqlite3.connect('search_data.db') as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS search_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                url TEXT,
                title TEXT,
                snippet TEXT,
                content TEXT,
                embedding BLOB,
                click_count INTEGER DEFAULT 0,
                result_type TEXT
            )
        ''')
        conn.commit()

# FAISS index management
def add_to_faiss_index(embedding: np.ndarray, doc_id: int):
    try:
        faiss_index.add_with_ids(np.array([embedding]).astype('float32'), np.array([doc_id]))
    except RuntimeError as e:
        print(f"Error adding to FAISS index: {e}")
        # 오류 발생 시 인덱스에 추가하지 않고 계속 진행

def search_faiss_index(query_embedding: np.ndarray, k: int = 10) -> List[int]:
    try:
        distances, indices = faiss_index.search(np.array([query_embedding]).astype('float32'), k)
        return indices[0].tolist()
    except RuntimeError as e:
        print(f"Error searching FAISS index: {e}")
        return []

# Enhanced search function
@lru_cache(maxsize=100)
def google_search(query, api_key, cse_id, num_results=10, search_type=None):
    try:
        service = build("customsearch", "v1", developerKey=api_key)
        params = {
            'q': query,
            'cx': cse_id,
            'num': num_results,
            'dateRestrict': 'd30'
        }
        
        if search_type == "image":
            params['searchType'] = "image"
        elif search_type == "video":
            params['q'] += " site:youtube.com"
        
        result = service.cse().list(**params).execute()

        # 로그 추가
        print(f"Google Search Results ({search_type or 'web'}): {result}")

        if search_type == "image":
            return [{"title": item.get('title', 'No title'),
                     "link": item.get('link', 'No link'),
                     "image": item.get('image', {}).get('thumbnailLink', ''),
                     "result_type": "image"} for item in result.get('items', [])]
        elif search_type == "video":
            return [{"title": item.get('title', 'No title'),
                     "link": item.get('link', 'No link'),
                     "snippet": item.get('snippet', 'No description'),
                     "thumbnail": item.get('pagemap', {}).get('thumbnail', [{'url': ''}])[0].get('url', ''),
                     "result_type": "video"} for item in result.get('items', [])]
        else:
            return [{"title": item.get('title', 'No title'), 
                     "snippet": item.get('snippet', 'No snippet'), 
                     "link": item.get('link', 'No link'), 
                     "image": item.get('pagemap', {}).get('cse_image', [{'src': ''}])[0].get('src', ''),
                     "result_type": "web"} for item in result.get('items', [])]
    except Exception as e:
        print(f"Google Search Error: {e}")
        return []


@lru_cache(maxsize=100)
async def naver_search(query, num_results=10, search_type=None):
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    
    if search_type == "image":
        url = f"https://openapi.naver.com/v1/search/image?query={query}&display={num_results}"
    elif search_type == "video":
        url = f"https://openapi.naver.com/v1/search/vclip?query={query}&display={num_results}"
    else:
        url = f"https://openapi.naver.com/v1/search/webkr.json?query={query}&display={num_results}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                result = await response.json()
                if search_type == "image":
                    return [{"title": item['title'], "link": item['link'], "image": item['thumbnail'], "result_type": "image"} for item in result.get('items', [])]
                elif search_type == "video":
                    return [{"title": item['title'], "link": item['link'], "snippet": item['description'], "thumbnail": item['thumbnail'], "result_type": "video"} for item in result.get('items', [])]
                else:
                    return [{"title": item['title'], "snippet": item['description'], "link": item['link'], "result_type": "web"} for item in result.get('items', [])]
    except Exception as e:
        print(f"Naver Search Error: {e}")
        return []

async def enhanced_search(query: str, num_results: int = 20) -> Dict[str, List[Dict]]:
    google_web_task = asyncio.to_thread(google_search, query, GOOGLE_API_KEY, CSE_ID, num_results=num_results)
    google_image_task = asyncio.to_thread(google_search, query, GOOGLE_API_KEY, CSE_ID, num_results=num_results, search_type="image")
    google_video_task = asyncio.to_thread(google_search, query, GOOGLE_API_KEY, CSE_ID, num_results=num_results, search_type="video")
    naver_web_task = naver_search(query, num_results=num_results)
    naver_image_task = naver_search(query, num_results=num_results, search_type="image")
    naver_video_task = naver_search(query, num_results=num_results, search_type="video")
    
    google_web, google_images, google_videos, naver_web, naver_images, naver_videos = await asyncio.gather(
        google_web_task, google_image_task, google_video_task, naver_web_task, naver_image_task, naver_video_task
    )
    
    web_results = google_web + naver_web
    image_results = google_images + naver_images
    video_results = google_videos + naver_videos
    
    processed_web_results = await fetch_and_process_content(web_results)
    
    return {
        "web": processed_web_results,
        "images": image_results,
        "videos": video_results
    }

# Content fetching and processing
async def fetch_content_async(url: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                return ' '.join(chunk for chunk in chunks if chunk)[:10000]
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return ""

async def fetch_and_process_content(search_results: List[Dict]) -> List[Dict]:
    tasks = [fetch_content_async(result['link']) for result in search_results]
    contents = await asyncio.gather(*tasks)
    
    for i, content in enumerate(contents):
        if content:
            search_results[i]['content'] = content
            search_results[i]['preprocessed_content'] = preprocess_text(content)
            search_results[i]['embedding'] = sentence_transformer.encode(search_results[i]['preprocessed_content'])
        else:
            search_results[i]['content'] = "No content available."
            search_results[i]['preprocessed_content'] = ""
            search_results[i]['embedding'] = np.zeros(d)
    
    return search_results

# Text preprocessing
def preprocess_text(text: str) -> str:
    tokens = okt.morphs(text)
    processed_tokens = [token for token in tokens if token not in korean_stopwords and token.isalnum()]
    return ' '.join(processed_tokens)

# Ranking system
def rank_results(query: str, search_results: List[Dict]) -> List[Dict]:
    query_embedding = sentence_transformer.encode([query])[0]
    
    for result in search_results:
        # Semantic similarity
        semantic_score = np.dot(query_embedding, result['embedding']) / (np.linalg.norm(query_embedding) * np.linalg.norm(result['embedding']))
        
        # TF-IDF score
        tfidf_matrix = tfidf_vectorizer.fit_transform([query, result['preprocessed_content']])
        tfidf_score = tfidf_matrix[0].dot(tfidf_matrix[1].T).toarray()[0][0]
        
        # Click-through rate
        click_weight = get_click_weight(result['link'])
        
        # Combined score
        result['score'] = (semantic_score * 0.4) + (tfidf_score * 0.3) + (click_weight * 0.3)
    
    return sorted(search_results, key=lambda x: x['score'], reverse=True)

# Database operations
def store_search_result(result: Dict):
    with sqlite3.connect('search_data.db') as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO search_results (query, url, title, snippet, content, embedding, result_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            result['query'], 
            result['link'], 
            result.get('title', 'No title'), 
            result.get('snippet', ''),  # Provide default value if 'snippet' is missing
            result.get('content', ''), 
            result.get('embedding', np.zeros(d)).tobytes(), 
            result.get('result_type', 'web')  # Provide default value for 'result_type'
        ))
        doc_id = c.lastrowid
        conn.commit()
    
    if result['result_type'] == 'web' and 'embedding' in result:
        add_to_faiss_index(result['embedding'], doc_id)

def get_click_weight(url: str) -> float:
    with sqlite3.connect('search_data.db') as conn:
        c = conn.cursor()
        c.execute('SELECT click_count FROM search_results WHERE url = ?', (url,))
        result = c.fetchone()
    click_count = result[0] if result else 0
    return 1 + (click_count * 0.1)

def increment_click_count(url: str):
    with sqlite3.connect('search_data.db') as conn:
        c = conn.cursor()
        c.execute('''
            UPDATE search_results
            SET click_count = click_count + 1
            WHERE url = ?
        ''', (url,))
        conn.commit()

# RAG implementation
def generate_rag_response(query: str, ranked_results: List[Dict], model_choice: str) -> str:
    context = "\n\n".join([f"Title: {result['title']}\nContent: {result['content'][:500]}" for result in ranked_results[:5]])
    
    if model_choice == 'ChatGPT':
        prompt = f"""Based on the following search results, provide a comprehensive and insightful answer to the query: "{query}"

Search Results:
{context}

Guidelines:
1. Summarize key points from the search results.
2. Explain the significance of the information and provide additional context.
3. Present conflicting viewpoints objectively, if any.
4. Offer insights or implications related to the topic.
5. Suggest areas for further exploration, if applicable.
6. Ensure your response is informative, engaging, and tailored to the user's query.
7. Cite sources using [1], [2], etc., corresponding to the search result order.
8. Provide a response that goes beyond simple summarization, offering a deeper understanding.
9. Include only information directly relevant to the user's question.
10. Support your answer with examples, statistics, or case studies when possible.
11. Explain complex concepts in an easy-to-understand manner.
12. At the end, suggest 2-3 related questions for further exploration.

Your entire response must be in Korean."""

        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an advanced AI assistant specializing in providing comprehensive and insightful answers based on search results."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2000,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in ChatGPT query: {e}")
            return "ChatGPT 응답 생성 중 오류가 발생했습니다. 다시 시도해주세요."
    else:
        prompt = f"""Query: {query}

Search Results:
{context}

Based on these search results, provide a comprehensive and insightful answer in Korean. Follow these guidelines:
1. Summarize key points from the search results.
2. Explain the significance of the information and provide additional context.
3. Present conflicting viewpoints objectively, if any.
4. Offer insights or implications related to the topic.
5. Suggest areas for further exploration, if applicable.
6. Ensure your response is informative, engaging, and tailored to the user's query.
7. Cite sources using [1], [2], etc., corresponding to the search result order.
8. Provide a response that goes beyond simple summarization, offering a deeper understanding.
9. Include only information directly relevant to the user's question.
10. Support your answer with examples, statistics, or case studies when possible.
11. Explain complex concepts in an easy-to-understand manner.
12. At the end, suggest 2-3 related questions for further exploration.

Your entire response must be in Korean."""

        try:
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error in Gemini query: {e}")
            return "Gemini 응답 생성 중 오류가 발생했습니다. 다시 시도해주세요."

@app.route('/record_click', methods=['POST'])
def record_click():
    data = request.get_json()
    url = data['url']
    increment_click_count(url)
    return '', 204

@app.route('/', methods=['GET', 'POST'])
async def index():
    if request.method == 'POST':
        query = request.form['query']
        model_choice = request.form.get('model', 'ChatGPT')

        search_results = await enhanced_search(query)
        
        for result_type, results in search_results.items():
            for result in results:
                result['query'] = query
                store_search_result(result)

        ranked_web_results = rank_results(query, search_results['web'])
        
        response = generate_rag_response(query, ranked_web_results, model_choice)

        display_web_results = [{
            'title': result['title'],
            'snippet': result['snippet'][:200] + '...' if len(result['snippet']) > 200 else result['snippet'],
            'link': result['link'],
            'score': result['score']
        } for result in ranked_web_results[:10]]
        print(search_results['images'])
        print('---------------------')
        print(search_results['videos'])
        return render_template('index.html', 
                               query=query, 
                               response=response, 
                               web_results=display_web_results, 
                               image_results=search_results['images'][:6],
                               video_results=search_results['videos'][:6],
                               model_choice=model_choice)

    return render_template('index.html')

if __name__ == '__main__':
    init_db()
    app.run(debug=True)