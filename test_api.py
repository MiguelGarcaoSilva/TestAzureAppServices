"""
Simple test script to verify the Flask app is working correctly
Run this after training the model with: python train_model.py
"""

import requests
import json

BASE_URL = 'http://localhost:5000'

def test_home():
    """Test home endpoint"""
    print("\n=== Testing GET / ===")
    response = requests.get(f'{BASE_URL}/')
    print(response.json())
    assert response.status_code == 200

def test_info():
    """Test info endpoint"""
    print("\n=== Testing GET /info ===")
    response = requests.get(f'{BASE_URL}/info')
    print(response.json())
    assert response.status_code == 200

def test_health():
    """Test health endpoint"""
    print("\n=== Testing GET /health ===")
    response = requests.get(f'{BASE_URL}/health')
    print(response.json())
    assert response.status_code == 200

def test_predict_setosa():
    """Test prediction for Setosa"""
    print("\n=== Testing POST /predict (Setosa) ===")
    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = requests.post(f'{BASE_URL}/predict', json=data)
    result = response.json()
    print(json.dumps(result, indent=2))
    assert response.status_code == 200
    assert result['prediction'] == 'setosa'

def test_predict_versicolor():
    """Test prediction for Versicolor"""
    print("\n=== Testing POST /predict (Versicolor) ===")
    data = {
        "sepal_length": 7.0,
        "sepal_width": 3.2,
        "petal_length": 4.7,
        "petal_width": 1.4
    }
    response = requests.post(f'{BASE_URL}/predict', json=data)
    result = response.json()
    print(json.dumps(result, indent=2))
    assert response.status_code == 200
    assert result['prediction'] == 'versicolor'

def test_predict_virginica():
    """Test prediction for Virginica"""
    print("\n=== Testing POST /predict (Virginica) ===")
    data = {
        "sepal_length": 6.3,
        "sepal_width": 3.3,
        "petal_length": 6.0,
        "petal_width": 2.5
    }
    response = requests.post(f'{BASE_URL}/predict', json=data)
    result = response.json()
    print(json.dumps(result, indent=2))
    assert response.status_code == 200
    assert result['prediction'] == 'virginica'

def test_predict_missing_field():
    """Test prediction with missing field"""
    print("\n=== Testing POST /predict (Missing field) ===")
    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4
        # Missing petal_width
    }
    response = requests.post(f'{BASE_URL}/predict', json=data)
    print(response.json())
    assert response.status_code == 400

if __name__ == '__main__':
    try:
        print("Starting tests...")
        test_home()
        test_info()
        test_health()
        test_predict_setosa()
        test_predict_versicolor()
        test_predict_virginica()
        test_predict_missing_field()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("Make sure the Flask app is running on http://localhost:5000")
