"""Idempotent token-aware ingestion. Existing vectors are retained for unchanged chunks."""
import json
import os
from collections import defaultdict
from pathlib import Path
from uuid import uuid4
from src.config import load_config, resolve_path, embedding_signature, validate_index
from src.documents import unique_docs, chunk_id


def write_manifest(directory, config, count, **extra):
    path = Path(directory) / 'index_manifest.json'
    manifest = {'version': uuid4().hex, 'embedding': embedding_signature(config), 'count': count, **extra}
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def sync_chunks(db, chunks):
    chunks = unique_docs(chunks)
    existing = set(db.get()['ids'])
    wanted = {chunk_id(doc) for doc in chunks}
    additions = [doc for doc in chunks if chunk_id(doc) not in existing]
    # Only remove stale records after all new vectors have been written successfully.
    for start in range(0, len(additions), 64):
        batch = additions[start:start + 64]
        db.add_documents(batch, ids=[chunk_id(doc) for doc in batch])
    retained = [doc for doc in chunks if chunk_id(doc) in existing]
    for start in range(0, len(retained), 256):
        batch = retained[start:start + 256]
        db._collection.update(ids=[chunk_id(doc) for doc in batch], metadatas=[doc.metadata for doc in batch])
    stale = sorted(existing - wanted)
    for start in range(0, len(stale), 256):
        db.delete(ids=stale[start:start + 256])
    return {'added': len(additions), 'deleted': len(stale), 'count': len(wanted)}


def split_documents(documents, embeddings, config):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    tokenizer = embeddings.client.tokenizer
    limit = min(config['retrieval']['chunk_size'], embeddings.client.max_seq_length - 2)
    overlap = min(config['retrieval']['chunk_overlap'], limit - 1)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=limit, chunk_overlap=overlap,
        length_function=lambda text: len(tokenizer.encode(text, add_special_tokens=False, verbose=False)),
        separators=['\n\n', '\n', '。', '！', '？', '；', '. ', ' ', ''], keep_separator=True)
    chunks = splitter.split_documents(documents)
    counters = defaultdict(int)
    for doc in chunks:
        doc.metadata['token_count'] = len(tokenizer.encode(doc.page_content, add_special_tokens=False, verbose=False))
        if doc.metadata['token_count'] > limit:
            raise ValueError('分块超过模型 token 上限，请检查 tokenizer 和分块配置')
        source = doc.metadata.get('source', 'unknown')
        doc.metadata['chunk_index'] = counters[source]
        counters[source] += 1
        doc.metadata['chunk_id'] = chunk_id(doc)
    return unique_docs(chunks)


def load_and_index_documents(directory_path=None, persist_dir=None, config=None):
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    from langchain_community.vectorstores import Chroma
    from src.components import load_embeddings
    config = config or load_config()
    directory = Path(directory_path or resolve_path(config['db']['docs_dir'])).resolve()
    target = Path(persist_dir or resolve_path(config['db']['chroma_persist_dir'])).resolve()
    config = {**config, 'db': {**config['db'], 'chroma_persist_dir': str(target)}}
    if not directory.is_dir():
        raise ValueError(f'知识库目录不存在: {directory}')
    files = sorted(path for path in directory.rglob('*') if path.suffix.lower() in ('.txt', '.pdf'))
    if not files:
        raise ValueError('知识库没有可读文件，保留现有索引')
    if (target / 'chroma.sqlite3').exists():
        validate_index(config)
    documents = []
    for path in files:
        loader = TextLoader(str(path), encoding='utf-8') if path.suffix.lower() == '.txt' else PyPDFLoader(str(path))
        loaded = loader.load()  # Parsing failure must not delete old documents.
        for doc in loaded:
            doc.metadata['source'] = path.relative_to(directory).as_posix()
        documents.extend(loaded)
    embeddings = load_embeddings(config)
    chunks = split_documents(documents, embeddings, config)
    if not chunks:
        raise ValueError('未提取到正文，保留现有索引')
    db = Chroma(persist_directory=str(target), embedding_function=embeddings)
    summary = sync_chunks(db, chunks)
    write_manifest(target, config, summary['count'], split_unit='tokens',
                   chunk_size=config['retrieval']['chunk_size'], chunk_overlap=config['retrieval']['chunk_overlap'])
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == '__main__':
    load_and_index_documents()
