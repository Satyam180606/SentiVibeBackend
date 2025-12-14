import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from groq import Groqapp = Flask(__name__)
CORS(app)

# --- Initialize Analyzers ---
vader_analyzer = SentimentIntensityAnalyzer()

# --- Initialize Groq (Llama) Client ---
# The API key will be set as an environment variable in Render
try:
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception as e:
    print(f"Failed to initialize Groq client: {e}")
    groq_client = None

def get_llm_sentiment(text_to_analyze):
    """
    Analyzes text using Groq's Llama 3 model to determine if the sentiment
    is 'Healthy' or 'Unhealthy'.
    """
    if not groq_client:
        return "LLM Analysis not available."

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis expert. Your task is to classify the user's text. "
                        "You must respond with only one of two possible phrases: 'Healthy sentiment' or 'Unhealthy sentiment'. "
                        "Do not add any other words, explanations, or punctuation."
                    )
                },
                {
                    "role": "user",
                    "content": text_to_analyze,
                }
            ],
            model="llama3-8b-8192",
            temperature=0.1, # Low temperature for deterministic classification
            max_tokens=10,   # Limit response length
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Groq API call failed: {e}")
        return "LLM analysis failed."

@app.route('/score', methods=['POST'])
def get_sentiment_score():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid input, "text" field is required.'}), 400

    text_to_analyze = data['text']

    # --- Run all analyses ---
    vader_scores = vader_analyzer.polarity_scores(text_to_analyze)
    blob = TextBlob(text_to_analyze)
    llm_result = get_llm_sentiment(text_to_analyze)

    # --- Format Response ---
    response = {
        'vader': vader_scores,
        'textblob': {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        },
        'llm_analysis': llm_result # Add the new LLM result
    }

    return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
