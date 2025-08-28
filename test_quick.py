
import requests
import time

def test_with_timeout():
    try:
        print("🔍 Starting research...")
        response = requests.post(
            "http://localhost:8000/research",
            json={"query": "What is artificial intelligence?"},
            timeout=90  # 90 second timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result['success']}")
            print(f"📄 Report length: {len(result.get('report', ''))}")
            print(f"🔗 Citations: {len(result.get('citations', []))}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - check server logs")
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_with_timeout()
