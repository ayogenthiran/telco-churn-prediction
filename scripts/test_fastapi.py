"""
Comprehensive FastAPI Endpoint Testing Script
=============================================

This script tests all FastAPI endpoints to ensure they work correctly:
1. Health check endpoint (GET /)
2. Prediction endpoint (POST /predict)
3. API documentation (GET /docs)
4. OpenAPI schema (GET /openapi.json)

Usage:
    python scripts/test_fastapi.py

Prerequisites:
    - FastAPI server must be running on http://127.0.0.1:8000
    - Start server with: uvicorn src.app.main:app --reload
"""

import requests
import sys
from typing import Dict, Any

# Base URL for the API
BASE_URL = "http://127.0.0.1:8000"

def test_health_check() -> bool:
    """Test the root health check endpoint."""
    print("\n🔍 Testing Health Check Endpoint (GET /)...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ok":
                print("✅ Health check passed")
                return True
            else:
                print(f"❌ Unexpected response: {data}")
                return False
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Is the server running?")
        print("   Start with: uvicorn src.app.main:app --reload")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_prediction_endpoint(sample_data: Dict[str, Any]) -> bool:
    """Test the prediction endpoint with sample data."""
    print("\n🔍 Testing Prediction Endpoint (POST /predict)...")
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=sample_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            if "prediction" in result:
                print(f"✅ Prediction successful: {result['prediction']}")
                return True
            elif "error" in result:
                print(f"❌ Prediction returned error: {result['error']}")
                return False
            else:
                print(f"❌ Unexpected response format: {result}")
                return False
        else:
            print(f"❌ Prediction failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_api_docs() -> bool:
    """Test that API documentation is accessible."""
    print("\n🔍 Testing API Documentation (GET /docs)...")
    try:
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code == 200:
            print("✅ API documentation accessible")
            return True
        else:
            print(f"❌ Documentation failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_openapi_schema() -> bool:
    """Test that OpenAPI schema is accessible."""
    print("\n🔍 Testing OpenAPI Schema (GET /openapi.json)...")
    try:
        response = requests.get(f"{BASE_URL}/openapi.json")
        if response.status_code == 200:
            schema = response.json()
            # Verify key components exist
            if "paths" in schema and "/predict" in schema["paths"]:
                print("✅ OpenAPI schema valid and includes /predict endpoint")
                return True
            else:
                print("❌ OpenAPI schema missing expected paths")
                return False
        else:
            print(f"❌ OpenAPI schema failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_invalid_data() -> bool:
    """Test that the API handles invalid data gracefully."""
    print("\n🔍 Testing Invalid Data Handling...")
    try:
        # Missing required field
        invalid_data = {
            "gender": "Male",
            # Missing SeniorCitizen and other required fields
        }
        response = requests.post(f"{BASE_URL}/predict", json=invalid_data)
        
        if response.status_code == 422:  # Validation error
            print("✅ API correctly rejects invalid data (422 Unprocessable Entity)")
            return True
        else:
            print(f"⚠️  Unexpected status for invalid data: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("FastAPI Endpoint Testing")
    print("=" * 60)
    
    # Sample customer data matching the API schema
    sample_data = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 350.75
    }
    
    # High churn risk example
    high_risk_data = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 85.0
    }
    
    # Low churn risk example
    low_risk_data = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 45.0,
        "TotalCharges": 2700.0
    }
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("API Documentation", test_api_docs),
        ("OpenAPI Schema", test_openapi_schema),
        ("Prediction (Sample Data)", lambda: test_prediction_endpoint(sample_data)),
        ("Prediction (High Risk)", lambda: test_prediction_endpoint(high_risk_data)),
        ("Prediction (Low Risk)", lambda: test_prediction_endpoint(low_risk_data)),
        ("Invalid Data Handling", test_invalid_data),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
