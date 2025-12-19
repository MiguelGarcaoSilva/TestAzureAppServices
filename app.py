from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the pre-trained model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load the dataset info for reference
data_path = os.path.join(os.path.dirname(__file__), 'data', 'iris.csv')
df = pd.read_csv(data_path)

@app.route('/')
def home():
    """Home page with web interface"""
    return render_template('index.html')

@app.route('/api')
def api_info():
    """API information endpoint"""
    return jsonify({
        'message': 'Welcome to the Iris Flower Classification API',
        'version': '1.0.0',
        'endpoints': {
            'predict': '/predict',
            'info': '/info'
        }
    })

@app.route('/info')
def info():
    """Dataset information endpoint"""
    return jsonify({
        'dataset': 'Iris Flower Dataset',
        'total_samples': len(df),
        'features': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
        'classes': ['setosa', 'versicolor', 'virginica'],
        'description': 'Predicts iris flower species based on measurements'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction from form or JSON"""
    try:
        # Check if it's a form submission or JSON
        if request.form:
            # Form submission from web interface
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])
            
            # Create features array
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            
            # Make prediction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            
            iris_classes = ['setosa', 'versicolor', 'virginica']
            
            # Prepare data for template
            result = {
                'prediction': iris_classes[prediction],
                'probabilities': {
                    iris_classes[i]: float(probability[i]) 
                    for i in range(len(iris_classes))
                },
                'input': {
                    'sepal_length': sepal_length,
                    'sepal_width': sepal_width,
                    'petal_length': petal_length,
                    'petal_width': petal_width
                }
            }
            
            return render_template('result.html', **result)
        else:
            # JSON API request
            data = request.get_json()
            
            # Validate input
            required_fields = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            if not all(field in data for field in required_fields):
                return jsonify({
                    'error': 'Missing required fields',
                    'required_fields': required_fields
                }), 400
            
            # Extract features
            features = np.array([[
                data['sepal_length'],
                data['sepal_width'],
                data['petal_length'],
                data['petal_width']
            ]])
            
            # Make prediction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            
            iris_classes = ['setosa', 'versicolor', 'virginica']
            
            return jsonify({
                'input': data,
                'prediction': iris_classes[prediction],
                'probabilities': {
                    iris_classes[i]: float(probability[i]) 
                    for i in range(len(iris_classes))
                }
            })
    
    except Exception as e:
        if request.form:
            return f"Error: {str(e)}", 400
        else:
            return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5002)))
