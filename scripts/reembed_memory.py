"""Re-embed correction records into an isolated memory bank; leave old vectors intact."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, fingerprint, embedding_signature, PROJECT_ROOT


def reembed_memory(config, embeddings, source=None):
    from langchain_community.vectorstores import Chroma
    from src.memory.gold_memory_bank import GoldMemoryBank
    source = Path(source or PROJECT_ROOT / 'gold_memory_db').resolve()
    target = PROJECT_ROOT / 'gold_memory_db' / fingerprint(embedding_signature(config))[:12]
    if source == target.resolve():
        raise ValueError('新旧记忆目录不能相同')
    if not (source / 'chroma.sqlite3').exists():
        return {'records': 0, 'added': 0, 'target': str(target)}
    old = Chroma(persist_directory=str(source), embedding_function=embeddings)
    records = {}
    # Read metadata only: never query or reuse embeddings from the old collection.
    for metadata in old.get(include=['metadatas'])['metadatas']:
        if metadata and metadata.get('mem_id'):
            records.setdefault(metadata['mem_id'], metadata)
    bank = GoldMemoryBank(persist_dir=str(target), embeddings=embeddings,
                          distance_threshold=config.get('memory', {}).get('distance_threshold', 0.65))
    existing = {m.get('migration_id') for m in bank.db.get(include=['metadatas'])['metadatas'] if m}
    added = 0
    for identity, metadata in records.items():
        migration_id = fingerprint([str(source), identity])
        if migration_id in existing:
            continue
        query, correct = metadata.get('original_query', ''), metadata.get('correct_answer', '')
        if not query or not correct:
            raise ValueError('旧记忆缺少问题或正确答案，停止迁移')
        # Stable IDs make an interrupted migration safe to retry.
        from langchain_core.documents import Document
        new_metadata = {**metadata, 'migration_id': migration_id}
        docs = [Document(page_content=text, metadata={**new_metadata, 'type': kind})
                for kind, text in [('query', query), ('answer', correct)]]
        bank.db.add_documents(docs, ids=[migration_id + '_query', migration_id + '_answer'])
        added += 1
    # Rebuild records.csv from the new bank, including records created since migration.
    import csv
    all_records = {}
    for m in bank.db.get(include=['metadatas'])['metadatas']:
        if m and m.get('mem_id'):
            all_records.setdefault(m['mem_id'], m)
    fields = ['mem_id', 'original_query', 'wrong_answer', 'correct_answer', 'user_id', 'category', 'source_file']
    temporary = bank.records_file.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_records.values())
    temporary.replace(bank.records_file)
    return {'records': len(records), 'added': added, 'target': str(target)}


if __name__ == '__main__':
    from src.components import load_embeddings
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--source')
    args = parser.parse_args()
    config = load_config(args.config)
    print(json.dumps(reembed_memory(config, load_embeddings(config), args.source), ensure_ascii=False))
