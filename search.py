from openai import OpenAI
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
load_dotenv()
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
CSE_ID=os.getenv('CSE_ID')
GOOGLE_API_KEY=os.getenv('GOOGLE_API')
client = OpenAI(api_key=OPENAI_API_KEY)
def google_search(query, api_key, cse_id):
    service = build("customsearch", "v1", developerKey=api_key)
    result = service.cse().list(q=query, cx=cse_id).execute()
    return result.get('items', [])

def chatgpt_query(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content":"You are an assistant that summarizes search results."},
            {"role": "user", "content":prompt},
        ],
        max_tokens=800,
    )
    content = response.choices[0].message.content.strip()
    return content

def search_gpt(query, google_api_key, cse_id):
    search_results = google_search(query, google_api_key, cse_id)
    
    if not search_results:
        return "No search results found."

    # 첫 번째 검색 결과의 설명을 가져옴
    search_summary = search_results[0].get('snippet', 'No summary available.')
    
    # ChatGPT에 검색 결과에 대한 요약 및 의견 요청
    prompt = f"Here is some information I found on the topic '{query}': {search_summary}. Can you provide a brief summary or opinion?"
    
    gpt_response = chatgpt_query(prompt)
    
    return {
        "search_summary": search_summary,
        "gpt_response": gpt_response
    }

# 예제 검색
query = "Python programming best practices"
result = search_gpt(query, GOOGLE_API_KEY, CSE_ID)

print("Search Summary:", result['search_summary'])
print("ChatGPT Response:", result['gpt_response'])