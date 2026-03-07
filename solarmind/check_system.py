import requests
import json

def check_system():
    # Check health
    health = requests.get("http://localhost:8000/health").json()
    print(f"Health: {json.dumps(health, indent=2)}")
    
    # Check inverters
    invs = requests.get("http://localhost:8000/inverters").json()
    print(f"Total Inverters in API: {len(invs)}")
    if invs:
        print(f"First Inverter Sample: {json.dumps(invs[0], indent=2)}")
        
    # Check a specific trend
    if invs:
        id = invs[0]['id']
        trends = requests.get(f"http://localhost:8000/inverters/{id}/trends").json()
        print(f"Trends for {id}: {len(trends)} points")
        if trends:
            print(f"First Trend Point: {json.dumps(trends[0], indent=2)}")

if __name__ == "__main__":
    try:
        check_system()
    except Exception as e:
        print(f"Error: {e}")
