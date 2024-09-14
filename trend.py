import os
from dotenv import load_dotenv
import openai
import google.generativeai as genai
from typing import List, Tuple, Dict
import json
import asyncio
import aiohttp

load_dotenv()

# Set up OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')


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
    print(' '.join(openai_keywords))
    return openai_keywords

queries = [
    "퍼플렉시티AI에 대해 알려줘",
    "GPT-4에 대해 알려줘",
    "이 코드가 어떻게 작동하는지 알려줘",
    "AI 기술이 발전하는 이유는 무엇인가요?",
    "파이썬으로 챗봇을 만드는 방법을 알려줘",
    "GPT-5에 대해 알려줘"
]

for query in queries:
    process_query(query)