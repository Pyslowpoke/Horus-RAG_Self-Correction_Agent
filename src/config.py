"""Shared configuration for indexing, evaluation and the application."""
from pathlib import Path
import hashlib
import json
import os
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    with Path(path or os.getenv('HORUS_CONFIG') or PROJECT_ROOT / "config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("config.yaml 必须是配置映射")
    retrieval = config["retrieval"]
    if not 0 <= retrieval["chunk_overlap"] < retrieval["chunk_size"]:
        raise ValueError("chunk_overlap 必须小于 chunk_size")
    if min(retrieval["top_k"], retrieval["candidate_k"], retrieval["rerank_k"]) < 1:
        raise ValueError("检索数量必须大于 0")
    return config


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def provider_config(config):
    return {**config, "api_key": os.getenv(config["api_key_env"], "")}


def embedding_signature(config):
    return {key: config["embedding"].get(key) for key in
            ("model_name", "normalize_embeddings", "query_prefix")}


def index_version(config):
    manifest = resolve_path(config["db"]["chroma_persist_dir"]) / "index_manifest.json"
    return fingerprint(manifest.read_text(encoding="utf-8")) if manifest.exists() else "unversioned"


def validate_index(config):
    manifest = resolve_path(config["db"]["chroma_persist_dir"]) / "index_manifest.json"
    if not manifest.exists():
        raise ValueError("索引缺少版本信息，请先运行 python -m src.data_ingestion 或迁移旧索引")
    stored = json.loads(manifest.read_text(encoding="utf-8"))
    if stored["embedding"] != embedding_signature(config):
        raise ValueError("Embedding 配置与索引不匹配，请选择新索引目录并重新入库")
