from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Backend selection
    llm_backend: Literal["openai", "ollama"] = "ollama"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"
    openai_temperature: float = 0.1
    openai_max_tokens: int = 2048

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_temperature: float = 0.1
    ollama_num_ctx: int = 4096

    # RAG
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 5
    bm25_weight: float = 0.4
    vector_weight: float = 0.6

    # Storage
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "finagent_docs"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # Agent
    agent_max_iterations: int = 10
    agent_verbose: bool = True

    @field_validator("chroma_persist_dir")
    @classmethod
    def ensure_chroma_dir(cls, v: str) -> str:
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

    @property
    def active_model(self) -> str:
        return self.openai_model if self.llm_backend == "openai" else self.ollama_model

    @property
    def active_embed_model(self) -> str:
        return self.openai_embed_model if self.llm_backend == "openai" else self.ollama_embed_model


def _merge_yaml_into_env(yaml_path: str = "config.yaml") -> None:
    """Load config.yaml values as env vars if not already set by .env."""
    if not Path(yaml_path).exists():
        return
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}

    mapping = {
        "llm_backend": data.get("llm_backend"),
        "openai_model": (data.get("openai") or {}).get("model"),
        "openai_embed_model": (data.get("openai") or {}).get("embed_model"),
        "openai_temperature": (data.get("openai") or {}).get("temperature"),
        "openai_max_tokens": (data.get("openai") or {}).get("max_tokens"),
        "ollama_base_url": (data.get("ollama") or {}).get("base_url"),
        "ollama_model": (data.get("ollama") or {}).get("model"),
        "ollama_embed_model": (data.get("ollama") or {}).get("embed_model"),
        "ollama_temperature": (data.get("ollama") or {}).get("temperature"),
        "ollama_num_ctx": (data.get("ollama") or {}).get("num_ctx"),
        "chunk_size": (data.get("rag") or {}).get("chunk_size"),
        "chunk_overlap": (data.get("rag") or {}).get("chunk_overlap"),
        "top_k_retrieval": (data.get("rag") or {}).get("top_k"),
        "bm25_weight": (data.get("rag") or {}).get("bm25_weight"),
        "vector_weight": (data.get("rag") or {}).get("vector_weight"),
        "chroma_persist_dir": (data.get("storage") or {}).get("chroma_persist_dir"),
        "chroma_collection_name": (data.get("storage") or {}).get("collection_name"),
        "api_host": (data.get("api") or {}).get("host"),
        "api_port": (data.get("api") or {}).get("port"),
        "log_level": (data.get("api") or {}).get("log_level"),
        "agent_max_iterations": (data.get("agent") or {}).get("max_iterations"),
        "agent_verbose": (data.get("agent") or {}).get("verbose"),
    }
    for key, val in mapping.items():
        if val is not None and key.upper() not in os.environ:
            os.environ[key.upper()] = str(val)


_merge_yaml_into_env()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
