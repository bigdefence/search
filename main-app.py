from flask import Flask, render_template, request, jsonify, stream_with_context, Response
from dotenv import load_dotenv
import asyncio
import json
from services.search import fetch_all_search_results, fetch_and_process_content, semantic_search
from utils.database import init_db, increment_click_count
from services.models import chatgpt_query, gemini_query
from utils.text_processing import process_query

load_dotenv()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        model_choice = request.form.get('model', 'ChatGPT')
        optimized_query = process_query(query)
        print(optimized_query)
        text_results, image_results, video_results = asyncio.run(fetch_all_search_results(optimized_query))
        processed_text_results = asyncio.run(fetch_and_process_content(text_results))

        top_indices = semantic_search(optimized_query, processed_text_results)
        prompt = f"Search results:\n"
        for i, index in enumerate(top_indices, 1):
            result = processed_text_results[index]
            result_text = (
                f"{i}. Title: {result.get('title', 'No Title')}\n"
                f"Source: {result.get('source', 'Unknown')}\n"
                f"Content: {result.get('content', 'No Content')[:2000]}\n\n"
            )
            prompt += result_text

        display_results = []
        for i, index in enumerate(top_indices[:8], 1):
            result = processed_text_results[index]
            display_results.append({
                'title': result.get('title', 'No Title'),
                'snippet': result.get('snippet', 'No Snippet')[:200] + '...' if len(result.get('snippet', '')) > 200 else result.get('snippet', 'No Snippet'),
                'link': result.get('link', 'No Link'),
            })

        image_results_processed = [
            {
                "link": image.get('link'),
                "thumbnail": image.get('image'),
                "title": image.get('title')
            } for image in image_results
        ]

        video_results_processed = [
            {
                "link": video.get('link'),
                "thumbnail": video.get('thumbnail'),
                "title": video.get('title')
            } for video in video_results
        ]

        def generate():
            ai_response = ""
            if model_choice == 'ChatGPT':
                for chunk in chatgpt_query(prompt, optimized_query):
                    ai_response += chunk
                    yield chunk
            elif model_choice == 'Gemini':
                for chunk in gemini_query(prompt, optimized_query):
                    ai_response += chunk
                    yield chunk

            final_response = {
                "image_results": image_results_processed,
                "video_results": video_results_processed,
                "search_results": display_results
            }
            # JSON 데이터를 별도로 전송
            yield json.dumps(final_response)


        return Response(stream_with_context(generate()), content_type='application/json')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
