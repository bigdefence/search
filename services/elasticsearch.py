from googleapiclient.discovery import build
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from readability import Document
import requests
from sentence_transformers import SentenceTransformer
from utils.config import GOOGLE_API_KEY, CSE_ID, NAVER_CLIENT_ID, NAVER_CLIENT_SECRET, YOUTUBE_API_KEY
from utils.database import get_click_count
from utils.text_processing import preprocess_text
from elasticsearch import AsyncElasticsearch, helpers, exceptions
from datetime import datetime

youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
sentence_transformer = SentenceTransformer('distilbert-base-nli-mean-tokens')

es = None

async def get_es_connection():
    global es
    for _ in range(3):  # Try connecting 3 times
        try:
            es = AsyncElasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}],
                                    request_timeout=30,
                                    max_retries=10,
                                    retry_on_timeout=True)
            if await es.ping():
                print("Elasticsearch에 연결되었습니다.")
                return es
        except exceptions.ConnectionError:
            print("연결 실패, 다시 시도 중...")
        await asyncio.sleep(1)  # 재시도 전 잠시 대기
    print("Elasticsearch 연결 실패")
    return None

async def initialize_es():
    global es
    es = await get_es_connection()

# 초기화 함수는 여기서 호출하지 않습니다.
# asyncio.run(initialize_es())

async def create_index_if_not_exists():
    index_name = "search_results"
    index_body = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "korean": {
                        "type": "custom",
                        "tokenizer": "nori_tokenizer",
                        "filter": ["lowercase", "nori_readingform", "nori_number"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "title": {"type": "text", "analyzer": "korean"},
                "content": {"type": "text", "analyzer": "korean"},
                "snippet": {"type": "text", "analyzer": "korean"},
                "link": {"type": "keyword"},
                "image": {"type": "keyword"},
                "source": {"type": "keyword"},
                "timestamp": {"type": "date"},
                "click_count": {"type": "integer"},
                "embedding": {"type": "dense_vector", "dims": 768}
            }
        }
    }
    try:
        if not await es.indices.exists(index=index_name):
            await es.indices.create(index=index_name, body=index_body)
            print(f"Index '{index_name}' created successfully")
        else:
            print(f"Index '{index_name}' already exists")
    except Exception as e:
        print(f"Error creating index: {e}")

async def index_search_results_async(search_results):
    global es
    if es is None:
        print("Elasticsearch 연결이 설정되지 않았습니다.")
        return
    try:
        actions = [{
            "_index": "search_results",
            "_id": result['link'],
            "_source": {
                **result,
                "timestamp": datetime.now().isoformat(),
                "click_count": get_click_count(result['link']),
                "embedding": sentence_transformer.encode(result.get('content', result.get('snippet', ''))).tolist()
            }
        } for result in search_results]
        
        await es.bulk(actions)
    except Exception as e:
        print(f"Elasticsearch indexing error: {e}")

async def elasticsearch_search(query, size=20):
    global es
    if es is None:
        print("Elasticsearch 연결이 설정되지 않았습니다.")
        return []
    try:
        query_embedding = sentence_transformer.encode(query).tolist()
        
        search_body = {
            "size": size,
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "should": [
                                {"match": {"title": query}},
                                {"match": {"content": query}},
                                {"match": {"snippet": query}}
                            ]
                        }
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + log1p(doc['click_count'].value) + 1.0 / (1.0 + doc['timestamp'].value.getMillis() - params.now)",
                        "params": {
                            "query_vector": query_embedding,
                            "now": datetime.now().timestamp() * 1000
                        }
                    }
                }
            }
        }
        
        response = await es.search(index="search_results", body=search_body)
        return [hit['_source'] for hit in response['hits']['hits']]
    except Exception as e:
        print(f"Elasticsearch search error: {e}")
        return []

async def fetch_search_results(search_func, *args, **kwargs):
    try:
        return await search_func(*args, **kwargs)
    except Exception as e:
        print(f"{search_func.__name__} Error: {e}")
        return []

async def fetch_all_search_results(query):
    num_results = 10
    search_tasks = [
        fetch_search_results(google_search, query, GOOGLE_API_KEY, CSE_ID, num_results),
        fetch_search_results(google_search, query, GOOGLE_API_KEY, CSE_ID, num_results, search_type="image"),
        fetch_search_results(naver_search, query, num_results),
        fetch_search_results(naver_search, query, num_results, search_type="image"),
        fetch_search_results(youtube_search, query)
    ]
    
    results = await asyncio.gather(*search_tasks)
    
    text_results = results[0] + results[2]
    image_results = results[1] + results[3]
    video_results = results[4]
    
    all_results = text_results + image_results + video_results
    
    await index_search_results_async(all_results)
    es_results = await elasticsearch_search(query)
    
    return es_results, image_results, video_results

async def fetch_content_async(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                html = await response.text()
                doc = Document(html)
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
    
    for result, content in zip(search_results, contents):
        if content:
            result['content'] = content
            result['preprocessed_content'] = preprocess_text(content)
        else:
            result['content'] = result.get('snippet', '')
            result['preprocessed_content'] = preprocess_text(result.get('snippet', ''))
    
    return search_results
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