import time
from rag.ingest import ingest_inverter_status, cleanup_old_documents
from rag.retriever import query

print("Ingesting dummy data into ChromaDB...")
now = int(time.time())
for i in range(5):
    # Simulate data over the last 5 days
    ts = now - (i * 86400)
    ingest_inverter_status(
        inverter_id="INV_001",
        plant_id="PLANT_1",
        block_id="B1",
        timestamp_unix=ts,
        risk_score=0.9 - (i * 0.1),  # Higher risk recently
        risk_level=["CRITICAL", "HIGH", "MEDIUM", "LOW", "LOW"][i],
        shap_top5=[
            {"feature": "thermal_gradient", "shap_value": 0.45},
            {"feature": "string_mismatch_cv", "shap_value": 0.22}
        ]
    )

print("Running cleanup (should not delete anything recent)...")
cleanup_old_documents()

print("Querying RAG: 'What is the recent status of INV_001?'")
results = query("What is the recent status of INV_001?", top_k=5)

print(f"Got {len(results)} results:")
for i, res in enumerate(results):
    meta = res["metadata"]
    print(f"[{i+1}] Score: {res['combined_score']:.3f} | Inverter: {meta.get('inverter_id')} | Risk: {meta.get('risk_score')}")

assert len(results) >= 3, f"Expected at least 3 results, got {len(results)}"
assert all(r["metadata"].get("inverter_id") is not None for r in results), "Null inverter_id citations found"
print("\nValidation passed!")
