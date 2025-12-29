#!/usr/bin/env python3
"""
Test script for DermaAI CNN + RAG integration
"""
import requests
from PIL import Image
import io
import time
import subprocess
import sys
import os

def start_server():
    """Start the server in background"""
    print("🚀 Starting DermaAI Server...")
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()

    process = subprocess.Popen([
        sys.executable, '-m', 'uvicorn',
        'app.main:app',
        '--host', '127.0.0.1',
        '--port', '8001'
    ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return process

def wait_for_server(max_attempts=30):
    """Wait for server to be ready"""
    print("⏳ Waiting for server to start...")
    for attempt in range(max_attempts):
        try:
            response = requests.get('http://127.0.0.1:8001/health', timeout=5)
            if response.status_code == 200:
                data = response.json()
                print("✅ Server is ready!")
                print(f"   Status: {data['status']}")
                print(f"   Models loaded: {data['models_loaded']}")
                print(f"   RAG available: {data['rag_available']}")
                return True
        except:
            pass

        time.sleep(2)
        print(f"   Attempt {attempt + 1}/{max_attempts}...")

    return False

def test_prediction():
    """Test the prediction endpoint"""
    print("\n🖼️  Creating test image...")
    # Create a simple test image (red square)
    img = Image.new('RGB', (224, 224), color='red')
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='JPEG')
    img_bytes = img_buffer.getvalue()

    print("🤖 Testing CNN + RAG prediction...")
    url = 'http://127.0.0.1:8001/predict'
    files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
    data = {'user_query': 'Is this skin cancer?'}

    try:
        response = requests.post(url, files=files, data=data, timeout=60)
        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("\n🎉 SUCCESS! CNN + RAG Integration Working!")
            print("=" * 60)
            print("📊 CNN RESULTS:")
            print(f"   Disease: {result.get('disease')}")
            print(f"   Confidence: {result.get('confidence'):.1%}")
            print(f"   Risk Level: {result.get('risk_level')}")
            print(f"   Medical Category: {result.get('medical_category')}")
            print(f"   Model Used: {result.get('model_used')}")

            print("\n🧠 RAG-GENERATED EXPLANATION:")
            print(f"   {result.get('explanation')}")

            print("\n👨‍⚕️ MEDICAL RECOMMENDATION:")
            print(f"   {result.get('recommendation')}")

            print("\n⚖️  MEDICAL DISCLAIMER:")
            print(f"   {result.get('disclaimer')}")

            print(f"\n😊 Avatar Emotion: {result.get('emotion')}")

            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def main():
    print("🧪 Testing DermaAI CNN + RAG Integration")
    print("=" * 50)

    # Start server
    server_process = start_server()

    try:
        # Wait for server
        if not wait_for_server():
            print("❌ Server failed to start properly")
            return

        # Test prediction
        success = test_prediction()

        if success:
            print("\n✅ Integration test PASSED!")
            print("Your CNN → RAG pipeline is working correctly!")
        else:
            print("\n❌ Integration test FAILED!")

    finally:
        # Clean up
        print("\n🧹 Stopping server...")
        server_process.terminate()
        server_process.wait()
        print("✅ Server stopped.")

if __name__ == "__main__":
    main()