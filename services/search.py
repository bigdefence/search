from googleapiclient.discovery import build
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from readability import Document
import requests
from rank_bm25 import BM25Okapi
import nltk
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.config import GOOGLE_API_KEY, CSE_ID, NAVER_CLIENT_ID, NAVER_CLIENT_SECRET, YOUTUBE_API_KEY
from utils.database import get_click_weight
from utils.text_processing import preprocess_text

youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
sentence_transformer = SentenceTransformer('distilbert-base-nli-mean-tokens')

async def google_search(query, api_key, cse_id, num_results=10, search_type=None):
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
        
        result = await asyncio.to_thread(service.cse().list(**params).execute)
        
        if search_type == "image":
            return [{"title": item.get('title', 'No title'),
                     "link": item.get('link', 'No link'),
                     "image": item.get('image', {}).get('thumbnailLink', '')} for item in result.get('items', [])]
        else:
            return [{"title": item.get('title', 'No title'), 
                     "snippet": item.get('snippet', 'No snippet'), 
                     "link": item.get('link', 'No link'), 
                     "image": item.get('pagemap', {}).get('cse_image', [{'src': ''}])[0].get('src', '')} for item in result.get('items', [])]
    except Exception as e:
        print(f"Google Search Error: {e}")
        return []

async def naver_search(query, num_results=10, search_type=None):
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    
    if search_type == "image":
        url = f"https://openapi.naver.com/v1/search/image?query={query}&display={num_results}&sort=date"
    else:
        url = f"https://openapi.naver.com/v1/search/webkr.json?query={query}&display={num_results}&sort=date"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                result = await response.json()
                if search_type == "image":
                    return [{"title": item['title'], "link": item['link'], "image": item['thumbnail']} for item in result.get('items', [])]
                else:
                    return [{"title": item['title'], "snippet": item['description'], "link": item['link'], "source": "Naver"} for item in result.get('items', [])]
    except Exception as e:
        print(f"Naver Search Error: {e}")
        return []
async def youtube_search(query, max_results=3):
    try:
        search_response = youtube.search().list(
            q=query,
            type='video',
            part='id,snippet',
            maxResults=max_results
        ).execute()

        videos = []
        for search_result in search_response.get('items', []):
            video = {
                "title": search_result['snippet']['title'],
                "link": f"https://www.youtube.com/watch?v={search_result['id']['videoId']}",
                "snippet": search_result['snippet']['description'],
                "thumbnail": search_result['snippet']['thumbnails']['medium']['url']
            }
            videos.append(video)
        return videos
    except Exception as e:
        print(f"YouTube Search Error: {e}")
        return []

async def fetch_all_search_results(query):
    num_results_google = 10
    num_results_naver = 10

    google_text_task = google_search(query, GOOGLE_API_KEY, CSE_ID, num_results=num_results_google)
    google_image_task = google_search(query, GOOGLE_API_KEY, CSE_ID, num_results=num_results_google, search_type="image")
    
    naver_text_task = naver_search(query, num_results=num_results_naver)
    naver_image_task = naver_search(query, num_results=num_results_naver, search_type="image")
    
    youtube_task = youtube_search(query)
    
    google_text, google_images, naver_text, naver_images, youtube_videos = await asyncio.gather(
        google_text_task, google_image_task,
        naver_text_task, naver_image_task,
        youtube_task
    )
    
    text_results = google_text + naver_text
    image_results = google_images + naver_images
    video_results = youtube_videos
    
    return text_results, image_results, video_results


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

def prepare_corpus_for_bm25(search_results):
    corpus = []
    for result in search_results:
        tokens = nltk.word_tokenize(result['preprocessed_content'])
        corpus.append(tokens)
    return corpus

def bm25_search_with_sentence_transformer(query, search_results, top_k=10):
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
    bm25_indices = bm25_search_with_sentence_transformer(query, search_results, top_k)
    faiss_indices = faiss_search(query, search_results, top_k)

    combined_indices = np.intersect1d(bm25_indices, faiss_indices)
    
    if len(combined_indices) < top_k:
        combined_indices = np.concatenate([combined_indices, bm25_indices, faiss_indices])[:top_k]
    
    return combined_indices

def semantic_search_with_click_data(query, search_results, top_k=10):
    combined_indices = combined_search(query, search_results, top_k)
    return combined_indices