from flask import Flask, render_template, request, jsonify, stream_with_context, Response
from dotenv import load_dotenv
import asyncio
import json
from services.search import fetch_all_search_results, fetch_and_process_content, semantic_search_with_click_data
from utils.database import init_db, increment_click_count
from services.models import chatgpt_query, gemini_query
from utils.text_processing import process_query

load_dotenv()

app = Flask(__name__)

@app.route('/record_click', methods=['POST'])
def record_click():
    data = request.get_json()
    url = data['url']
    increment_click_count(url)
    return '', 204

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        model_choice = request.form.get('model', 'ChatGPT')
        optimized_query = process_query(query)
        print(optimized_query)
        text_results, image_results, video_results = asyncio.run(fetch_all_search_results(optimized_query))
        processed_text_results = asyncio.run(fetch_and_process_content(text_results))

        top_indices = semantic_search_with_click_data(optimized_query,processed_text_results)
        
        prompt = f"Search results:\n"
        for i, index in enumerate(top_indices, 1):
            result = processed_text_results[index]
            result_text = (
                f"{i}. Title: {result.get('title', 'No Title')}\n"
                f"Summary: {result.get('snippet', 'No Summary')}\n"
                f"Source: {result.get('source', 'Unknown')}\n"
                f"Content: {result.get('content', 'No Content')[:2000]}\n\n"
            )
            prompt += result_text

        display_results = []
        for i, index in enumerate(top_indices[:10], 1):
            result = processed_text_results[index]
            display_results.append({
                'title': result.get('title', 'No Title'),
                'snippet': result.get('snippet', 'No Snippet')[:200] + '...' if len(result.get('snippet', '')) > 200 else result.get('snippet', 'No Snippet'),
                'link': result.get('link', 'No Link'),
            })

        def generate():
            ai_response = ""
            if model_choice == 'ChatGPT':
                for chunk in chatgpt_query(prompt, optimized_query):
                    ai_response += chunk
                    yield chunk
            else:
                for chunk in gemini_query(prompt, optimized_query):
                    ai_response += chunk
                    yield chunk
            
            final_response = {
                "ai_response": ai_response,
                "search_results": display_results,
                "image_results": image_results,
                "video_results": video_results
            }
            yield "\n" + json.dumps(final_response)

        return Response(stream_with_context(generate()), content_type='application/json')

    return render_template('index.html')

if __name__ == '__main__':
    init_db()
    app.run(debug=True)