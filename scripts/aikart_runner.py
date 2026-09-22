import json
import os
import sys
from pathlib import Path

# Ensure root workspace is on python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.agents.graph import run_agent
from src.rag.indexer import get_document_count, seed_sample_documents


def main():
    # 1. Ensure sample documents are loaded so out-of-the-box queries work immediately
    if get_document_count() == 0:
        seed_sample_documents()

    # 2. Read AIKART_INPUT from environment
    raw_input = os.environ.get("AIKART_INPUT", "{}")
    try:
        input_data = json.loads(raw_input)
    except Exception:
        input_data = {"query": str(raw_input)}

    query = (
        input_data.get("query")
        or input_data.get("topic")
        or input_data.get("question")
        or "Summarize key financial metrics and revenue growth from the financial report."
    )
    analysis_type = input_data.get("analysis_type", "Comprehensive Financial Analysis")

    full_prompt = query
    if analysis_type and analysis_type != "Comprehensive Financial Analysis":
        full_prompt = f"[{analysis_type}] {query}"

    # 3. Execute agent
    try:
        res = run_agent(full_prompt)
        answer = res.get("answer", "No answer generated.")
        sources = res.get("sources", [])
        intent = res.get("intent", "financial_analysis")

        # Format markdown response
        response_md = f"{answer}\n\n"
        if sources:
            response_md += f"**Grounded Sources:** {', '.join(sources)}\n"
        if intent:
            response_md += f"*(Identified Intent: {intent})*\n"

        output_data = {
            "format": "markdown",
            "response": response_md,
        }
    except Exception as e:
        output_data = {
            "format": "markdown",
            "response": f"### Error running FinSight AI\n\n`{str(e)}`",
        }

    # 4. Write to /aikart/output.json (or local fallback if running outside container)
    output_path = Path("/aikart/output.json")
    if not output_path.parent.exists():
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            output_path = ROOT_DIR / "aikart_output.json"

    output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    print(f"Output successfully written to {output_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
