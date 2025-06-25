from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import urllib.parse
import re
import numpy as np
from collections import Counter
import pandas as pd
import joblib  # Alternative to pickle for ML models
import tldextract

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter web requests

# Multiple approaches to load the model
model = None

# Load top-1m domains
with open("top-1m.csv", "r") as file:
    top_domains = set(line.strip().split(',')[1] for line in file if ',' in line)

def is_trusted_domain(domain):
    return 1 if domain in top_domains else 0


def load_model_with_fallbacks():
    """Try multiple methods to load the model"""
    global model
    
    # Method 1: Try with joblib (recommended for sklearn models)
    try:
        model = joblib.load('safelink_malicious_url_rf_model.pkl')
        print("Model loaded successfully with joblib!")
        return True
    except Exception as e:
        print(f"Joblib loading failed: {e}")
    
    # Method 2: Try with pickle using different protocols
    for protocol in [None, 0, 1, 2, 3, 4, 5]:
        try:
            with open('safelink_malicious_url_rf_model.pkl', 'rb') as f:
                if protocol is None:
                    model = pickle.load(f)
                else:
                    # This won't work for loading, but shows the concept
                    model = pickle.load(f)
            print(f"Model loaded successfully with pickle (protocol {protocol})!")
            return True
        except Exception as e:
            print(f"Pickle loading failed with protocol {protocol}: {e}")
            continue
    
    # Method 3: Try loading with different encoding
    try:
        with open('safelink_malicious_url_rf_model.pkl', 'rb') as f:
            model = pickle.load(f, encoding='latin1')
        print("Model loaded successfully with latin1 encoding!")
        return True
    except Exception as e:
        print(f"Latin1 encoding failed: {e}")
    
    # Method 4: Try loading with bytes encoding
    try:
        with open('safelink_malicious_url_rf_model.pkl', 'rb') as f:
            model = pickle.load(f, encoding='bytes')
        print("Model loaded successfully with bytes encoding!")
        return True
    except Exception as e:
        print(f"Bytes encoding failed: {e}")
    
    print("All model loading methods failed!")
    return False

# Try to load the model
load_model_with_fallbacks()

def extract_url_features(url):
    """Extract features from URL - same as your training function"""
    features = {}
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_underscores'] = url.count('_')
    features['num_slashes'] = url.count('/')
    features['num_questionmarks'] = url.count('?')
    features['num_equal'] = url.count('=')
    features['num_at'] = url.count('@')
    features['num_and'] = url.count('&')
    features['num_exclamation'] = url.count('!')
    features['num_space'] = url.count(' ')
    features['num_tilde'] = url.count('~')
    features['num_comma'] = url.count(',')
    features['num_plus'] = url.count('+')
    features['num_asterisk'] = url.count('*')
    features['num_hashtag'] = url.count('#')
    features['num_dollar'] = url.count('$')
    features['num_percent'] = url.count('%')
    
    try:
        parsed = urllib.parse.urlparse(url)
        features['scheme_length'] = len(parsed.scheme)
        features['netloc_length'] = len(parsed.netloc)
        features['path_length'] = len(parsed.path)
        features['query_length'] = len(parsed.query)
        features['fragment_length'] = len(parsed.fragment)
        #domain = parsed.netloc
        ext = tldextract.extract(url)
        domain = ext.domain + '.' + ext.suffix if ext.suffix else ''
        features['domain_length'] = len(domain)
        features['subdomain_count'] = domain.count('.') 
        features['is_ip'] = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+', domain) else 0
    except:
        features.update({
            'scheme_length': 0, 'netloc_length': 0, 'path_length': 0,
            'query_length': 0, 'fragment_length': 0, 'domain_length': 0,
            'subdomain_count': 0, 'is_ip': 0
        })

    features['is_top1m_domain'] = is_trusted_domain(domain)
    
    suspicious_words = ['login', 'verify', 'account', 'update', 'secure', 'banking', 
                        'suspended', 'limited', 'click', 'confirm', 'immediately']
    #features['suspicious_words'] = sum(1 for word in suspicious_words if word in url.lower())
    suspicious_count = sum(1 for word in suspicious_words if word in url.lower())

    # Prevent trusted domains from being penalized
    if features['is_top1m_domain']:
        suspicious_count = 0

    features['suspicious_words'] = suspicious_count

    if len(url) > 0:
        char_counts = Counter(url)
        entropy = -sum((count/len(url)) * np.log2(count/len(url)) for count in char_counts.values())
        features['entropy'] = entropy
    else:
        features['entropy'] = 0

    features['digit_ratio'] = sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0
    features['letter_ratio'] = sum(c.isalpha() for c in url) / len(url) if len(url) > 0 else 0
    
    return features

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get URL from request
        data = request.get_json()
        url = data.get('url', '')
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Extract features
        features = extract_url_features(url)
        
        # Convert to DataFrame (important: maintain same order as training)
        feature_names = [
            'url_length', 'num_dots', 'num_hyphens', 'num_underscores', 'num_slashes',
            'num_questionmarks', 'num_equal', 'num_at', 'num_and', 'num_exclamation',
            'num_space', 'num_tilde', 'num_comma', 'num_plus', 'num_asterisk',
            'num_hashtag', 'num_dollar', 'num_percent', 'scheme_length', 'netloc_length',
            'path_length', 'query_length', 'fragment_length', 'domain_length',
            'subdomain_count', 'is_ip', 'is_top1m_domain', 'suspicious_words',
            'entropy', 'digit_ratio', 'letter_ratio'
            ]

        
        # Create feature vector in correct order
        feature_vector = [[features[name] for name in feature_names]]
        
        # Make prediction with error handling
        try:
            prediction = model.predict(feature_vector)[0]
            probabilities = model.predict_proba(feature_vector)[0]
        except Exception as pred_error:
            return jsonify({'error': f'Prediction error: {str(pred_error)}'}), 500
        
        # # Assuming 1 = malicious, 0 = benign
        # result = {
        #     'url': url,
        #     'prediction': int(prediction),
        #     'is_malicious': bool(prediction),
        #     'confidence': float(max(probability)),
        #     'malicious_probability': float(probability[1] if len(probability) > 1 else probability[0]),
        #     'features': features
        # }
        class_labels = {0: 'benign', 1: 'defacement', 2: 'malware', 3: 'phishing'}
        
        result = {
            'url': url,
            'predicted_class': int(prediction),
            'predicted_label': class_labels.get(int(prediction), 'unknown'),
            'confidence': float(np.max(probabilities)),
            'class_probabilities': {
                class_labels[i]: float(prob) for i, prob in enumerate(probabilities)
            },
            'features': features
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Malicious URL Detection API',
        'version': '1.0',
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST)',
            'reload_model': '/reload_model (POST)'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/reload_model', methods=['POST'])
def reload_model():
    """Endpoint to try reloading the model"""
    success = load_model_with_fallbacks()
    return jsonify({'success': success, 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)