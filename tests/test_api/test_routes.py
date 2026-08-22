from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "llm_backend" in data
    assert "document_chunks" in data


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "FinAgent" in response.text


def test_api_info_endpoint():
    response = client.get("/api/info")
    assert response.status_code == 200
    data = response.json()
    assert "FinAgent" in data["name"]
    assert "endpoints" in data


def test_list_documents_empty():
    response = client.get("/documents/")
    assert response.status_code == 200
    data = response.json()
    assert "sources" in data
    assert "total_chunks" in data


@patch("src.api.routes.documents.ingest_file")
def test_ingest_document_txt(mock_ingest):
    mock_ingest.return_value = 5
    file_content = b"Apple revenue was $383.3 billion in 2023."
    response = client.post(
        "/documents/ingest",
        files={"file": ("test_report.txt", io.BytesIO(file_content), "text/plain")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["nodes_indexed"] == 5
    assert "test_report.txt" in data["filename"]


def test_ingest_unsupported_file_type():
    response = client.post(
        "/documents/ingest",
        files={"file": ("report.xlsx", io.BytesIO(b"fake"), "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]


@patch("src.rag.indexer.delete_source")
def test_delete_document(mock_delete):
    mock_delete.return_value = 3
    response = client.delete("/documents/test_file.txt")
    assert response.status_code == 200
    data = response.json()
    assert data["deleted_chunks"] == 3


@patch("src.rag.indexer.clear_all_documents")
def test_clear_documents(mock_clear):
    mock_clear.return_value = 10
    response = client.post("/documents/clear")
    assert response.status_code == 200
    data = response.json()
    assert data["cleared_chunks"] == 10


def test_calculate_ratio():
    response = client.post("/query/ratio", json={"numerator": 100.0, "denominator": 50.0, "ratio_name": "P/E"})
    assert response.status_code == 200
    data = response.json()
    assert data["value"] == 2.0
    assert data["ratio_name"] == "P/E"


@patch("src.api.routes.query.get_document_count")
@patch("src.api.routes.query.run_agent")
def test_query_endpoint(mock_run_agent, mock_count):
    mock_count.return_value = 10
    mock_run_agent.return_value = {
        "answer": "Apple's revenue was $383.3 billion.",
        "sources": ["apple.txt"],
        "intent": "simple_qa",
        "agent_trace": ["[planner] classified as 'simple_qa'", "[qa] generated answer"],
    }

    response = client.post("/query/", json={"query": "What was Apple revenue?"})
    assert response.status_code == 200
    data = response.json()
    assert "383.3 billion" in data["answer"]
    assert data["intent"] == "simple_qa"


@patch("src.api.routes.query.get_document_count")
def test_query_fails_when_no_docs(mock_count):
    mock_count.return_value = 0
    response = client.post("/query/", json={"query": "What was Apple revenue?"})
    assert response.status_code == 400
    assert "No documents" in response.json()["detail"]

