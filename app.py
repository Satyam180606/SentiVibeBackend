from flask import Flask, request, jsonify
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import os # Import the os module

app = Flask(__name__)
CORS(app)

vader_analyzer = SentimentIntensityAnalyzer()

@app.route('/score', methods=['POST'])
def get_sentiment_score():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid input, "text" field is required.'}), 400

    text_to_analyze = data['text']

    # VADER Analysis
    vader_scores = vader_analyzer.polarity_scores(text_to_analyze)

    # TextBlob Analysis
    blob = TextBlob(text_to_analyze)
    textblob_scores = {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity
    }

    response = {
        'vader': vader_scores,
        'textblob': textblob_scores
    }

    return jsonify(response)

# This part is updated for production
if __name__ == '__main__':
    # Render provides the PORT environment variable.
    # The default is 5001 for local testing.
    port = int(os.environ.get('PORT', 5001))
    # For production, debug should be False and host should be 0.0.0.0
    app.run(host='0.0.0.0', port=port, debug=False)