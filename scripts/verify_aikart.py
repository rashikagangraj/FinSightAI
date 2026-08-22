import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.rag.indexer import seed_sample_documents
from src.agents.graph import run_agent

print("Seeding sample documents...")
seed_sample_documents()

queries = [
    "What was AIKART's revenue in FY2025–26?",
    "What was AIKART's net profit?",
    "What was AIKART's cash balance?",
    "Compare AIKART and Tesla revenue"
]

for q in queries:
    res = run_agent(q)
    print("=" * 60)
    print("QUERY: " + q)
    print("INTENT: " + str(res.get("intent")))
    print("ANSWER:\n" + str(res.get("answer")))
    print("SOURCES: " + str(res.get("sources")))
    print("=" * 60)
