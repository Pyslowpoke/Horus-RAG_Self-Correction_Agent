"""Model construction shared by app, ingestion and offline evaluation."""
import logging
from threading import Lock
from src.config import validate_index, resolve_path, provider_config
from src.documents import read_documents, tokenize


def load_embeddings(config):
    from langchain_community.embeddings import HuggingFaceEmbeddings
    import torch
    torch.set_num_threads(config['reranker']['cpu_threads'])
    opts = config['embedding']
    embedding = HuggingFaceEmbeddings(model_name=opts['model_name'],
        model_kwargs={'device': opts['device'], 'local_files_only': opts['local_files_only']},
        encode_kwargs={'normalize_embeddings': opts['normalize_embeddings']})
    prefix = opts.get('query_prefix', '')
    if not prefix:
        return embedding
    from langchain_core.embeddings import Embeddings
    class QueryPrefixEmbedding(Embeddings):
        client = embedding.client
        def embed_documents(self, texts):
            return embedding.embed_documents(texts)
        def embed_query(self, text):
            return embedding.embed_query(prefix + text)
    return QueryPrefixEmbedding()


def load_db(config, embeddings):
    from langchain_community.vectorstores import Chroma
    validate_index(config)
    return Chroma(persist_directory=str(resolve_path(config['db']['chroma_persist_dir'])),
                  embedding_function=embeddings)


def build_bm25(db):
    from rank_bm25 import BM25Okapi
    docs = read_documents(db)
    return (BM25Okapi([tokenize(doc.page_content) or ['__empty__'] for doc in docs]) if docs else None), docs


def load_reranker(config):
    opts = config['reranker']
    if not opts['enabled']:
        return None
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(opts['cpu_threads'])
    device = opts['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        tokenizer = AutoTokenizer.from_pretrained(opts['model_name'], local_files_only=opts['local_files_only'])
        model = AutoModelForSequenceClassification.from_pretrained(opts['model_name'],
            local_files_only=opts['local_files_only']).to(device).eval()
        return {'tokenizer': tokenizer, 'model': model, 'device': device, 'lock': Lock(),
                'batch_size': opts['batch_size'], 'max_length': opts['max_length']}
    except Exception:
        logging.getLogger(__name__).exception('无法加载重排模型，启用词项过滤降级')
        return None


def make_retriever(config, db, bm25, docs, reranker):
    from src.retrievers.hybrid_retriever import HybridRetriever
    return HybridRetriever(db, bm25, docs, reranker_model=reranker,
                           min_rerank_score=config['reranker']['min_score'], **config['retrieval'])


def make_llm(config, light=False):
    from src.generators.llm_client import FaultTolerantLLM
    primary = config['llm']['light' if light else 'primary']
    fallback = config['llm']['primary' if light else 'fallback']
    return FaultTolerantLLM(provider_config(primary), provider_config(fallback),
        max_retries=config['llm']['max_retries'], max_tokens=config['generation']['max_tokens'],
        temperature=config['generation']['temperature'])
