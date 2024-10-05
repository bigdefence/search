from konlpy.tag import Okt
from stopwords import get_stopwords
import openai
import google.generativeai as genai
from datetime import datetime
from utils.config import GEMINI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-pro-002')
okt = Okt()
korean_stopwords = set(get_stopwords('ko'))

def preprocess_text(text):
    tokens = okt.morphs(text)
    processed_tokens = [token for token in tokens if token not in korean_stopwords and token.isalnum()]
    return ' '.join(processed_tokens)

# def extract_keywords_openai(query):
#     current_year = datetime.now().year
#     prompt = f"""
#     Transform the given user query into a concise, SEO-optimized search query by following these rules:
#     1. Remove all unnecessary words, including polite expressions, questions, and filler words.
#     2. Extract and keep only the essential keywords and phrases.
#     3. Maintain the original order of key concepts when possible.
#     4. For price-related queries, keep the price and put it before related terms like "추천" (recommendation).
#     5. Maintain the original language (Korean or English) of keywords.

#     Input query: "{query}"
#     Output only the optimized query without any additional text or explanations.
#     """
#     try:
#         response = openai.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=30,
#             temperature=0.7
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         print(f"OpenAI Error: {e}")
#         return ""

def extract_keywords_openai(query,model="chatgpt"):
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_year = datetime.now().year
    
    prompt = f"""
    Task: Transform the given conversational query into a concise, effective search query.
    Original query: "{query}"
    Current date: {current_date}

    Guidelines:
    1. Remove conversational elements like "알려줘", "추천해주세요", "알고 싶어요" etc.
    2. Keep core concepts, proper nouns, and specific terms intact (e.g., GPT-3, 퍼플렉시티AI).
    3. Preserve numbers and units when relevant (e.g., 10만원 이하).
    4. Include the current year ({current_year}) for queries about recent or upcoming events/trends.
    5. Maintain original language (Korean or English) of key terms.
    6. Abbreviate long phrases while keeping meaning (e.g., "맛있는 파스타 레스토랑" to "파스타 레스토랑").
    7. Keep location names if specified (e.g., 서울, 속초).
    8. Reflect search intent (e.g., 추천, 최신 소식).
    9. Limit output to 2-5 essential keywords or short phrases.
    10. Arrange keywords in order of importance.

    Output: Space-separated keywords forming a search query. No explanations or additional text.
    """
    
    try:
        if model.lower() == "chatgpt":
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a precise query optimization AI."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=25,
                temperature=0.8,
                top_p=0.9,
                frequency_penalty=0.2,
                presence_penalty=0.2
            )
            return response.choices[0].message.content.strip()
        elif model.lower() == "gemini":
            response = gemini_model.generate_content(prompt)
            return response.text.strip()
        else:
            raise ValueError("Invalid model specified. Choose 'chatgpt' or 'gemini'.")
    except Exception as e:
        print(f"API Error ({model}): {e}")
        return query

def clean_keywords(keywords):
    if isinstance(keywords, tuple):
        keywords = keywords[0] 
    keyword_list = list(dict.fromkeys(keywords.split(',')))
    return [kw.strip() for kw in keyword_list if len(kw.strip()) > 1]

def extract_keywords(query):
    openai_keywords = extract_keywords_openai(query,model='gemini')
    return clean_keywords(openai_keywords)

def process_query(query):
    openai_keywords = extract_keywords(query)
    return ' '.join(openai_keywords)