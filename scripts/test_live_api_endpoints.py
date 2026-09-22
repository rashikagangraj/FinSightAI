import json
import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.api.main import app
from src.core.config import get_settings
from src.rag.indexer import seed_sample_documents

def test_all_endpoints():
    print("=" * 70)
    print("STARTING LIVE API ENDPOINTS TEST SUITE")
    print("=" * 70)

    cfg = get_settings()
    auth_headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
    client = TestClient(app, headers=auth_headers)

    # 1. GET /health
    print("\n[1] Testing GET /health ...")
    res = client.get("/health")
    print(f"Status Code: {res.status_code}")
    print(f"Response: {json.dumps(res.json(), indent=2)}")
    assert res.status_code == 200

    # 2. GET /api/info
    print("\n[2] Testing GET /api/info ...")
    res = client.get("/api/info")
    print(f"Status Code: {res.status_code}")
    print(f"Response: {json.dumps(res.json(), indent=2)}")
    assert res.status_code == 200

    # 3. GET / (Dashboard HTML)
    print("\n[3] Testing GET / (Dashboard) ...")
    res = client.get("/")
    print(f"Status Code: {res.status_code}")
    print(f"Content-Type: {res.headers.get('content-type')}")
    print(f"Body Preview: {res.text[:120]}...")
    assert res.status_code == 200

    # 4. Ingest/Seed Documents
    print("\n[4] Ensuring Seed Documents are Loaded ...")
    seed_sample_documents()

    # 5. GET /documents/
    print("\n[5] Testing GET /documents/ ...")
    res = client.get("/documents/")
    print(f"Status Code: {res.status_code}")
    print(f"Response: {json.dumps(res.json(), indent=2)}")
    assert res.status_code == 200

    # 6. POST /query/ratio (Financial Ratio Calculation)
    print("\n[6] Testing POST /query/ratio (P/E & Profitability) ...")
    ratio_payload = {
        "numerator": 78000000.0,
        "denominator": 5700000.0,
        "ratio_name": "Revenue to Net Profit"
    }
    res = client.post("/query/ratio", json=ratio_payload)
    print(f"Status Code: {res.status_code}")
    print(f"Response: {json.dumps(res.json(), indent=2)}")
    assert res.status_code == 200

    # 7. POST /query/ (Agent Q&A with RAG Grounding)
    print("\n[7] Testing POST /query/ (Agent Q&A with Grounding) ...")
    query_payload = {
        "query": "What was AIKART's total revenue and net profit in FY2025-26?"
    }
    res = client.post("/query/", json=query_payload)
    print(f"Status Code: {res.status_code}")
    print(f"Response: {json.dumps(res.json(), indent=2)}")
    assert res.status_code == 200

    # 8. GET /cases/ (Case Studies Engine)
    print("\n[8] Testing GET /cases/ ...")
    res = client.get("/cases/")
    print(f"Status Code: {res.status_code}")
    cases_data = res.json()
    print(f"Loaded {len(cases_data)} interactive case studies")
    print(f"Sample Case: {json.dumps(cases_data[0], indent=2)}")
    assert res.status_code == 200

    print("\n" + "=" * 70)
    print("ALL API ENDPOINTS TESTED AND VERIFIED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    test_all_endpoints()
