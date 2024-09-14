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
from concurrent.futures import ThreadPoolExecutor
import requests
from readability import Document
from rank_bm25 import BM25Okapi
import nltk
import faiss
from pytrends.request import TrendReq
nltk.download('punkt')
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
sentence_transformer = SentenceTransformer('distilbert-base-nli-mean-tokens')

# Initialize the SQLite database
def init_db():
    with sqlite3.connect('click_data.db') as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS clicks (
                url TEXT PRIMARY KEY,  
                click_count INTEGER DEFAULT 0  
            )
        ''')
        conn.commit()

app = Flask(__name__)
def get_google_trends():
    pytrends = TrendReq(hl='ko-KR', tz=540)  # tz=540 is for KST (UTC+9)
    trending_searches = pytrends.trending_searches(pn='south_korea')
    trending_searches =trending_searches.values.tolist()[:10]
    flattened_list = [item for sublist in trending_searches for item in sublist]
    return flattened_list # Return top 10 trends
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
        
        if search_type == "image":
            return [{"title": item.get('title', 'No title'),
                     "link": item.get('link', 'No link'),
                     "image": item.get('image', {}).get('thumbnailLink', '')} for item in result.get('items', [])]
        elif search_type == "video":
            return [{"title": item.get('title', 'No title'),
                     "link": item.get('link', 'No link'),
                     "snippet": item.get('snippet', 'No description'),
                     "thumbnail": item.get('pagemap', {}).get('thumbnail', [{'url': ''}])[0].get('url', '')} for item in result.get('items', [])]
        else:
            return [{"title": item.get('title', 'No title'), 
                     "snippet": item.get('snippet', 'No snippet'), 
                     "link": item.get('link', 'No link'), 
                     "image": item.get('pagemap', {}).get('cse_image', [{'src': ''}])[0].get('src', '')} for item in result.get('items', [])]
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
                    return [{"title": item['title'], "link": item['link'], "image": item['thumbnail']} for item in result.get('items', [])]
                elif search_type == "video":
                    return [{"title": item['title'], "link": item['link'], "snippet": item['description'], "thumbnail": item['thumbnail']} for item in result.get('items', [])]
                else:
                    return [{"title": item['title'], "snippet": item['description'], "link": item['link'], "source": "Naver"} for item in result.get('items', [])]
    except Exception as e:
        print(f"Naver Search Error: {e}")
        return []

async def fetch_all_search_results(query):
    num_results_google = 10
    num_results_naver = 10

    google_text_task = asyncio.to_thread(google_search, query, GOOGLE_API_KEY, CSE_ID, num_results=num_results_google)
    naver_text_task = naver_search(query, num_results=num_results_naver)
    naver_image_task = naver_search(query, num_results=num_results_naver, search_type="image")
    naver_video_task = naver_search(query, num_results=num_results_naver, search_type="video")
    
    google_text, naver_text, naver_images, naver_videos = await asyncio.gather(
        google_text_task, naver_text_task, naver_image_task, naver_video_task
    )
    text_results = google_text + naver_text
    image_results = naver_images
    video_results = naver_videos
    
    return text_results, image_results, video_results

def preprocess_text(text):
    tokens = okt.morphs(text)
    processed_tokens = [token for token in tokens if token not in korean_stopwords and token.isalnum()]
    return ' '.join(processed_tokens)

async def fetch_content_async(url):
    try:
        # requests를 사용하여 웹 페이지 콘텐츠 가져오기
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # 오류 발생 시 예외 발생

        # readability를 사용하여 기사 본문 추출
        doc = Document(response.text)
        article_content = doc.summary()

        soup = BeautifulSoup(article_content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split(" "))
        return ' '.join(chunk for chunk in chunks if chunk)[:5000]

    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return ""

async def fetch_and_process_content(search_results):
    tasks = [fetch_content_async(result['link']) for result in search_results]
    contents = await asyncio.gather(*tasks)
    
    for i, content in enumerate(contents):
        if content:
            search_results[i]['content'] = content
            search_results[i]['preprocessed_content'] = preprocess_text(content)
        else:
            search_results[i]['content'] = "No content available."
            search_results[i]['preprocessed_content'] = ""
    
    return search_results

def increment_click_count(url):
    with sqlite3.connect('click_data.db') as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO clicks (url, click_count) 
            VALUES (?, 1) 
            ON CONFLICT(url) DO UPDATE SET click_count = click_count + 1
        ''', (url,))
        conn.commit()

def get_click_count(url):
    with sqlite3.connect('click_data.db') as conn:
        c = conn.cursor()
        c.execute('SELECT click_count FROM clicks WHERE url = ?', (url,))
        result = c.fetchone()
    return result[0] if result else 0

def get_click_weight(url):
    click_count = get_click_count(url)
    return 1 + (click_count * 0.1)

def prepare_corpus_for_bm25(search_results):
    corpus = []
    for result in search_results:
        tokens = nltk.word_tokenize(result['preprocessed_content'])
        corpus.append(tokens)
    return corpus

def bm25_search(query, search_results, top_k=10):
    corpus = prepare_corpus_for_bm25(search_results)
    query_tokens = nltk.word_tokenize(query)
    
    bm25 = BM25Okapi(corpus)
    doc_scores = bm25.get_scores(query_tokens)
    
    weighted_scores = [score * get_click_weight(result['link']) for score, result in zip(doc_scores, search_results)]
    top_indices = np.argsort(weighted_scores)[::-1][:top_k]
    return top_indices

def faiss_search(query, search_results, top_k=10):
    query_embedding = sentence_transformer.encode([query])
    document_embeddings = sentence_transformer.encode([result['preprocessed_content'] for result in search_results])
    
    d = query_embedding.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(document_embeddings))
    
    distances, indices = index.search(np.array(query_embedding), top_k)
    return indices[0]

def combined_search(query, search_results, top_k=10):
    bm25_indices = bm25_search(query, search_results, top_k)
    faiss_indices = faiss_search(query, search_results, top_k)

    combined_indices = np.intersect1d(bm25_indices, faiss_indices)
    
    if len(combined_indices) < top_k:
        combined_indices = np.concatenate([combined_indices, bm25_indices, faiss_indices])[:top_k]
    
    return combined_indices

def semantic_search_with_click_data(query, search_results, top_k=10):
    combined_indices = combined_search(query, search_results, top_k)
    return combined_indices

def chatgpt_query(prompt, query):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """You are an advanced AI assistant tasked with providing comprehensive and insightful answers based on search results. Analyze the given information, synthesize it, and present a well-structured response in Korean. Follow these guidelines:

1. Summarize the key points from the search results.
2. Explain the significance of the information and provide additional context or insights valuable to the user.
3. Present conflicting viewpoints objectively, if any.
4. Offer insights or implications related to the topic.
5. Suggest areas for further exploration, if applicable.
6. Ensure your response is informative, engaging, and tailored to the user's query.
7. Always cite your sources using [1], [2], etc., corresponding to the search result numbers.
8. Aim to provide a response that goes beyond simple summarization, offering a deeper understanding of the topic.
9. Include only information directly relevant to the user's question, excluding unnecessary details.
10. Support your answer with real-world examples, statistics, or case studies when possible.
11. Explain complex concepts in an easy-to-understand manner.
12. At the end of your response, suggest 1-2 related questions the user might want to explore further.

Remember, your entire response must be in Korean, regardless of the language of the search results or the user's query.

User query: {query}
"""},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in ChatGPT query: {e}")
        return "ChatGPT 응답 생성 중 오류가 발생했습니다. 다시 시도해주세요."

def gemini_query(prompt, query):
    try:
        system_prompt = f"""You are an advanced AI assistant tasked with providing comprehensive and insightful answers based on search results. Analyze the given information, synthesize it, and present a well-structured response in Korean. Follow these guidelines:

1. Concisely summarize the key points from the search results.
2. Explain the significance of the information and provide additional context or background information.
3. Present conflicting viewpoints objectively, if any, and explain the pros and cons of each perspective.
4. Offer insights or potential implications related to the topic.
5. Suggest areas for further exploration or related topics, if applicable.
6. Ensure your response is informative, engaging, and directly relevant to the user's query.
7. Always cite your sources using [1], [2], etc., corresponding to the search result numbers.
8. Go beyond simple summarization to provide a deep understanding and analysis of the topic.
9. Support your answer with real-world examples, statistics, or case studies when possible.
10. Explain complex concepts in an easy-to-understand manner, using analogies or examples if necessary.
11. Structure your response clearly, using subheadings to separate information if needed.
12. At the end of your response, suggest 2-3 related questions the user might want to explore further.

Remember, your entire response must be in Korean, regardless of the language of the search results or the user's query.

User query: {query}

Based on the above guidelines, provide a comprehensive and insightful answer in Korean using the given search results."""

        full_prompt = f"{system_prompt}\n\nSearch results:\n{prompt}"
        response = gemini_model.generate_content(full_prompt)
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

def extract_keywords_openai(query):
    prompt = f"""
    Extract 3-5 important keywords or short phrases from the following query. Focus only on technical terms, names of technologies, AI models, or core concepts related to AI. Ignore common words.
    
    Query: "{query}"
    Output only the keywords, separated by commas.
    Keywords:
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI Error: {e}")
        return ""


def clean_keywords(keywords):
    if isinstance(keywords, tuple):
        keywords = keywords[0] 
    keyword_list = list(dict.fromkeys(keywords.split(',')))
    return [kw.strip() for kw in keyword_list if len(kw.strip()) > 1]

def extract_keywords(query):
    openai_keywords = extract_keywords_openai(query)
    return clean_keywords(openai_keywords)

def process_query(query):
    openai_keywords = extract_keywords(query)
    return ' '.join(openai_keywords) 
@app.route('/fetch_trends')
def fetch_trends():
    try:
        google_trends = get_google_trends()
        def create_google_search_url(query):
            return f"https://www.google.com/search?q={query}"
        return {
            'google_trends': [{'title': trend, 'url': create_google_search_url(trend)} for trend in google_trends],
        }
    except Exception as e:
        print(f"Error fetching trends: {e}")
        return {
            'google_trends': [],
        }
@app.route('/', methods=['GET', 'POST'])
async def index():
    if request.method == 'POST':
        query = request.form['query']
        model_choice = request.form.get('model', 'ChatGPT')  # 모델 선택 기본값을 ChatGPT로 설정
        optimized_query=process_query(query)
        print(optimized_query)
        # 검색 결과 처리
        text_results, image_results, video_results = await fetch_all_search_results(optimized_query)
        processed_text_results = await fetch_and_process_content(text_results)
        print(video_results)  # 디버깅용 출력

        # 상위 검색 결과의 인덱스 추출
        top_indices = semantic_search_with_click_data(optimized_query, processed_text_results)
        
        # 모델에 전달할 프롬프트 생성
        prompt = f"Search results:\n"
        for i, index in enumerate(top_indices, 1):
            result = processed_text_results[index]
            result_text = (
                f"{i}. Title: {result.get('title', 'No Title')}\n"
                f"Summary: {result.get('snippet', 'No Summary')}\n"
                f"Source: {result.get('source', 'Unknown')}\n"
                f"Content: {result.get('content', 'No Content')[:500]}\n\n"
            )
            prompt += result_text

        # 모델 응답 생성
        try:
            if model_choice == 'ChatGPT':
                response = chatgpt_query(prompt, optimized_query)  # ChatGPT 모델 호출
            else:
                response = gemini_query(prompt, optimized_query)  # Gemini 모델 호출
        except Exception as e:
            print(f"Error generating response: {e}")
            response = "응답 생성 중 오류가 발생했습니다. 다시 시도해주세요."

        google_trends = get_google_trends()
        display_results = []
        for i, index in enumerate(top_indices[:10], 1):  # 상위 10개 결과만 표시
            result = processed_text_results[index]
            display_results.append({
                'title': result.get('title', 'No Title'),
                'snippet': result.get('snippet', 'No Snippet')[:200] + '...' if len(result.get('snippet', '')) > 200 else result.get('snippet', 'No Snippet'),
                'link': result.get('link', 'No Link'),
            })

        # 검색 결과와 응답을 템플릿에 전달
        return render_template(
            'index.html',
            query=query,
            response=response,
            search_results=display_results,
            image_results=image_results,
            video_results=video_results,
            model_choice=model_choice,
            google_trends=google_trends
        )

    return render_template('index.html')


if __name__ == '__main__':
    init_db()
    app.run(debug=True)