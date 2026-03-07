import requests

def test_query():
    url = "http://localhost:8000/query"
    payload = {
        "question": "What should I do about high temperatures in Inverter 001?",
        "session_id": "test_session"
    }
    # We need authentication? The route has Depends(get_current_user)
    # Let's get a token first.
    auth_url = "http://localhost:8000/auth/token"
    auth_payload = {"username": "admin", "password": "admin"}
    
    try:
        token_res = requests.post(auth_url, data=auth_payload)
        token = token_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        print("Sending query to SolarMind AI...")
        res = requests.post(url, json=payload, headers=headers)
        print(f"Status: {res.status_code}")
        print(f"Response: {res.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_query()
