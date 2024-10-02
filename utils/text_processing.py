from konlpy.tag import Okt
from stopwords import get_stopwords
import openai
from datetime import datetime

okt = Okt()
korean_stopwords = set(get_stopwords('ko'))

def preprocess_text(text):
    tokens = okt.morphs(text)
    processed_tokens = [token for token in tokens if token not in korean_stopwords and token.isalnum()]
    return ' '.join(processed_tokens)

def extract_keywords_openai(query):
    current_year = datetime.now().year
    prompt = f"""
    Transform the given user query into a concise, SEO-optimized search query by following these rules:
    1. Remove all unnecessary words, including polite expressions, questions, and filler words.
    2. Extract and keep only the essential keywords and phrases.
    3. Maintain the original order of key concepts when possible.
    4. For price-related queries, keep the price and put it before related terms like "추천" (recommendation).
    5. Maintain the original language (Korean or English) of keywords.

    Input query: "{query}"
    Output only the optimized query without any additional text or explanations.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
            temperature=0.1
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