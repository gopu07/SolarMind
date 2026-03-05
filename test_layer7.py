import os
# Force non-interactive test mode for DBs
os.environ["CHROMA_PERSIST_DIR"] = "./rag_test_chroma"

from fastapi.testclient import TestClient
from api.main import app
import sys

client = TestClient(app)

print("Fetching all registered routes:")
routes = [route.path for route in app.routes]
for r in sorted(set(routes)):
    print(" ", r)

expected_routes = ["/health", "/predict", "/predict/batch", "/query", "/auth/token", "/ws/stream/{plant_id}"]
missing = [r for r in expected_routes if r not in routes]

if missing:
    print("MISSING ROUTES:", missing)
    sys.exit(1)
else:
    print("All 5+ main endpoint routes are registered.")

print("\nTesting GET /health...")
response = client.get("/health")
print("Status Code:", response.status_code)
print("Response JSON:", response.json())

if response.status_code in [200, 207]:
    print("Health endpoint validated successfully (returns 200 or 207 due to missing key).")
else:
    print("Health check failed!")
    sys.exit(1)
