"""Copy old index to a backup, then migrate unique chunks and existing embeddings.
No model download, no remote API, no deletion/modification of the original index.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sqlite3
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, resolve_path, embedding_signature
from src.documents import chunk_id
from src.data_ingestion import write_manifest
from langchain_core.documents import Document


def migrate(source, target, config):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError('源目录和目标目录不能相同或互相包含')
    if target.exists():
        raise ValueError('迁移目标必须不存在，避免覆盖已有索引')
    if not (source / 'chroma.sqlite3').exists():
        raise ValueError('源索引不存在')
    source_manifest = source / 'index_manifest.json'
    source_embedding = (json.loads(source_manifest.read_text(encoding='utf-8'))['embedding']
                        if source_manifest.exists() else {'model_name': 'all-MiniLM-L6-v2',
                            'normalize_embeddings': False, 'query_prefix': ''})
    if embedding_signature(config) != source_embedding:
        raise ValueError('迁移不能更换 Embedding；新模型请从原始文档重新入库')
    backup = source.parent / 'backups' / ('chroma_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    shutil.copytree(source, backup)
    # Consistent SQLite backup, including committed WAL content.
    with sqlite3.connect((source / 'chroma.sqlite3').as_uri() + '?mode=ro', uri=True) as old:
        with sqlite3.connect(backup / 'chroma.sqlite3') as new:
            old.backup(new)
    import chromadb
    from chromadb.config import Settings
    settings = Settings(anonymized_telemetry=False)
    client = chromadb.PersistentClient(path=str(backup), settings=settings)
    collection = client.get_collection('langchain', embedding_function=None)
    data = collection.get(include=['documents', 'metadatas', 'embeddings'])
    unique = {}
    for text, metadata, vector in zip(data['documents'], data['metadatas'], data['embeddings']):
        metadata = dict(metadata or {})
        # Canonicalize old absolute sources relative to the knowledge directory when possible.
        old_source = str(metadata.get('source', 'unknown')).replace('\\', '/')
        docs_dir = resolve_path(config['db']['docs_dir']).resolve().as_posix()
        metadata['source'] = old_source[len(docs_dir) + 1:] if old_source.startswith(docs_dir + '/') else old_source
        doc = Document(page_content=text, metadata=metadata)
        identity = chunk_id(doc)
        metadata['chunk_id'] = identity
        unique.setdefault(identity, (text, metadata, vector))
    dest_client = chromadb.PersistentClient(path=str(target), settings=settings)
    dest = dest_client.create_collection('langchain', embedding_function=None,
                                          metadata=collection.metadata)
    ids = list(unique)
    for start in range(0, len(ids), 128):
        batch = ids[start:start + 128]
        dest.add(ids=batch, documents=[unique[i][0] for i in batch],
                 metadatas=[unique[i][1] for i in batch], embeddings=[unique[i][2] for i in batch])
    if dest.count() != len(unique):
        raise RuntimeError('迁移数量核验失败')
    write_manifest(target, config, len(unique), migrated_from=str(source), split_unit='legacy_characters')
    return {'original': len(data['ids']), 'unique': len(unique), 'backup': str(backup), 'target': str(target)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='chroma_db')
    parser.add_argument('--target')
    args = parser.parse_args()
    config = load_config()
    print(json.dumps(migrate(resolve_path(args.source), args.target or resolve_path(config['db']['chroma_persist_dir']), config), ensure_ascii=False, indent=2))
