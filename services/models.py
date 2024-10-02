import openai
import google.generativeai as genai
from utils.config import OPENAI_API_KEY, GEMINI_API_KEY

openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-pro-002')

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
            max_tokens=3000,
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        print(f"Error in ChatGPT query: {e}")
        yield "ChatGPT 응답 생성 중 오류가 발생했습니다. 다시 시도해주세요."

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
        response = gemini_model.generate_content(full_prompt,stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        print(f"Error in Gemini query: {e}")
        yield "Gemini 응답 생성 중 오류가 발생했습니다. 다시 시도해주세요."