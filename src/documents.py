"""Stable chunk identity and consistent lexical normalization."""
import hashlib
import json
import re
import jieba
from langchain_core.documents import Document

STOP_WORDS = set("的 了 是 在 和 有 这 那 我 你 他 她 它 什么 怎么 如何 为什么 可以 能 会 吗 呢 吧 请 请问 一下 介绍 a the is are was were what how why can do".split())


def tokenize(text):
    return [word for word in jieba.lcut(text.lower())
            if word.strip() and word not in STOP_WORDS and re.search(r"\w", word)]


def chunk_id(doc):
    if doc.metadata.get("chunk_id"):
        return doc.metadata["chunk_id"]
    identity = [str(doc.metadata.get("url") or doc.metadata.get("source", "unknown")).replace("\\", "/"),
                doc.metadata.get("page", ""), doc.page_content.strip()]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()


def unique_docs(docs):
    seen = set()
    result = []
    for doc in docs:
        identity = chunk_id(doc)
        if identity not in seen:
            seen.add(identity)
            result.append(Document(page_content=doc.page_content,
                                   metadata={**doc.metadata, "chunk_id": identity}))
    return result


def read_documents(db):
    result = db.get(include=["documents", "metadatas"])
    return unique_docs([Document(page_content=text, metadata={**(meta or {}), "storage_id": identity})
                        for identity, text, meta in zip(result["ids"], result["documents"], result["metadatas"])])


def lexical_relevance(query, doc):
    terms = set(tokenize(query))
    if not terms:
        return 0.0
    return len(terms.intersection(tokenize(doc.page_content))) / len(terms)
