"""
黄金记忆库模块

功能：记忆存储、向量检索、偏好注入、文件监控

文件格式（correction_inbox/*.txt）：
    Q: 原始问题
    Wrong: 错误答案
    Correct: 修正答案
"""

import os
import uuid
import time
from pathlib import Path
from typing import List, Dict, Set
from dataclasses import dataclass

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


@dataclass
class CorrectionMemory:
    mem_id: str
    original_query: str
    wrong_answer: str
    correct_answer: str
    user_id: str
    category: str
    source_file: str


class GoldMemoryBank:
    """
    用法：
        bank = GoldMemoryBank()
        bank.add_memory("什么是RAG", "RAG是...", "RAG是检索增强生成...")
        results = bank.retrieve_preferences("RAG技术", top_k=3)
    """

    def __init__(self, persist_dir=None, embedding_model="all-MiniLM-L6-v2", device="cpu", embeddings=None):
        if persist_dir is None:
            project_root = Path(__file__).parent.parent.parent
            persist_dir = str(project_root / "gold_memory_db")

        self.persist_dir = persist_dir
        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name=embedding_model, model_kwargs={"device": device}
        )
        self.db = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
        self.records_file = Path(persist_dir) / "records.csv"
        self._ensure_records_file()
        self._observer = None

    def _ensure_records_file(self):
        if not self.records_file.exists():
            self.records_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.records_file, "w", encoding="utf-8") as f:
                f.write("mem_id,original_query,wrong_answer,correct_answer,user_id,category,source_file\n")

    def _is_duplicate(self, source_file, original_query):
        if not self.records_file.exists():
            return False
        with open(self.records_file, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                if source_file in line and original_query[:20] in line:
                    return True
        return False

    def add_memory(self, original_query, wrong_answer, correct_answer,
                   user_id="default", category="correction", source_file="") -> str:
        mem_id = str(uuid.uuid4())[:8]

        query_doc = Document(
            page_content=original_query,
            metadata={"mem_id": mem_id, "type": "query", "original_query": original_query,
                      "wrong_answer": wrong_answer, "correct_answer": correct_answer,
                      "user_id": user_id, "category": category, "source_file": source_file},
        )
        answer_doc = Document(
            page_content=correct_answer,
            metadata={"mem_id": mem_id, "type": "answer", "original_query": original_query,
                      "wrong_answer": wrong_answer, "correct_answer": correct_answer,
                      "user_id": user_id, "category": category, "source_file": source_file},
        )

        self.db.add_documents([query_doc, answer_doc])
        self.db.persist()

        with open(self.records_file, "a", encoding="utf-8") as f:
            q = original_query.replace(",", "，").replace("\n", " ")
            w = wrong_answer.replace(",", "，").replace("\n", " ")
            c = correct_answer.replace(",", "，").replace("\n", " ")
            f.write(f"{mem_id},{q},{w},{c},{user_id},{category},{source_file}\n")

        return mem_id

    def retrieve_preferences(self, query, top_k=3, distance_threshold=0.65) -> List[Dict]:
        results = self.db.similarity_search_with_score(query, k=top_k * 2)
        seen_ids: Set[str] = set()
        memories = []

        for doc, score in results:
            mem_id = doc.metadata.get("mem_id", "")
            if mem_id in seen_ids or score > distance_threshold:
                continue
            seen_ids.add(mem_id)
            memories.append({
                "mem_id": mem_id,
                "original_query": doc.metadata.get("original_query", ""),
                "wrong_answer": doc.metadata.get("wrong_answer", ""),
                "correct_answer": doc.metadata.get("correct_answer", ""),
                "distance": round(score, 3),
            })
            if len(memories) >= top_k:
                break

        return memories

    def get_memory_context_prompt(self, query, top_k=3) -> str:
        memories = self.retrieve_preferences(query, top_k=top_k)
        if not memories:
            return ""
        lines = ["以下是用户的历史纠偏记录，请在回答时参考："]
        for i, mem in enumerate(memories, 1):
            lines.append(f"\n{i}. 用户问：{mem['original_query']}")
            lines.append(f"   错误回答：{mem['wrong_answer']}")
            lines.append(f"   正确回答：{mem['correct_answer']}")
        return "\n".join(lines)

    def parse_correction_file(self, file_path) -> List[Dict]:
        records, current = [], {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("Q:"):
                    if current.get("query") and current.get("correct"):
                        records.append(current)
                    current = {"query": line[2:].strip()}
                elif line.startswith("Wrong:"):
                    current["wrong"] = line[6:].strip()
                elif line.startswith("Correct:"):
                    current["correct"] = line[8:].strip()
        if current.get("query") and current.get("correct"):
            records.append(current)
        return records

    def import_correction_file(self, file_path, force=False) -> int:
        records = self.parse_correction_file(file_path)
        source_file = Path(file_path).name
        count = 0
        for record in records:
            query = record.get("query", "")
            if not query or (not force and self._is_duplicate(source_file, query)):
                continue
            self.add_memory(
                original_query=query, wrong_answer=record.get("wrong", ""),
                correct_answer=record.get("correct", ""), source_file=source_file,
            )
            count += 1
        return count

    def get_memory_count(self) -> int:
        try:
            with open(self.records_file, "r", encoding="utf-8") as f:
                return max(0, sum(1 for _ in f) - 1)
        except Exception:
            return 0

    def get_document_count(self) -> int:
        try:
            return self.db._collection.count()
        except Exception:
            return 0

    def __len__(self):
        return self.get_memory_count()

    def close(self):
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=2)
            self._observer = None


# ============ 文件监控 ============

class CorrectionFileHandler:
    def __init__(self, memory_bank):
        self.memory_bank = memory_bank
        self._processed_files: Set[str] = set()

    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith(".txt"):
            return
        file_path = event.src_path
        if file_path in self._processed_files:
            return
        self._processed_files.add(file_path)
        try:
            count = self.memory_bank.import_correction_file(file_path)
            if count > 0:
                print(f"[GoldMemory] 导入 {Path(file_path).name}，{count} 条记忆")
        except Exception as e:
            print(f"[GoldMemory] 导入失败: {e}")
        finally:
            self._processed_files.discard(file_path)


def start_watcher(memory_bank, watch_dir=None):
    from watchdog.observers import Observer
    if watch_dir is None:
        watch_dir = str(Path(__file__).parent.parent.parent / "data" / "correction_inbox")
    Path(watch_dir).mkdir(parents=True, exist_ok=True)

    handler = CorrectionFileHandler(memory_bank)
    observer = Observer()
    observer.schedule(handler, watch_dir, recursive=False)
    observer.daemon = True
    observer.start()
    memory_bank._observer = observer
    return observer


def stop_watcher(observer):
    if observer:
        observer.stop()
        observer.join(timeout=2)


def import_all_corrections(memory_bank, watch_dir=None):
    if watch_dir is None:
        watch_dir = str(Path(__file__).parent.parent.parent / "data" / "correction_inbox")
    watch_path = Path(watch_dir)
    if not watch_path.exists():
        return
    total = 0
    for txt_file in watch_path.glob("*.txt"):
        try:
            count = memory_bank.import_correction_file(str(txt_file))
            total += count
        except Exception:
            pass
    print(f"[GoldMemory] 批量导入完成，共 {total} 条记忆")
