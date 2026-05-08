# src/adapters/web/flask_app.py
import os
from flask import Flask, request, jsonify, render_template
from src.core.entities import Email
from src.use_cases.phishing_analyzer import PhishingAnalyzer

def create_app(analyzer: PhishingAnalyzer) -> Flask:
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    app = Flask(__name__, template_folder=template_dir)

    @app.route('/', methods=['GET'])
    def index():
        return render_template('index.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.json
        if not data or 'email_text' not in data:
            return jsonify({"error": "Missing 'email_text' payload in request body"}), 400
            
        email = Email(raw_text=data['email_text'])
        
        try:
            result = analyzer.analyze(email)
            response = {
                "is_phishing": result.is_phishing,
                "is_promotional": result.is_promotional,
                "label": result.label,
                "label_name": result.label_name,
                "confidence": round(result.confidence, 4),
                "explanations": result.explanations
            }
            return jsonify(response), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({"status": "healthy"}), 200

    return app
