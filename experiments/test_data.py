# test_data.py
# 测试数据集

TEST_CASES = [
    {
        "query": "什么是RAG？",
        "expected_answer": "RAG是检索增强生成技术，通过在生成前检索外部知识库来提高答案质量。",
        "expected_claims": ["RAG", "检索增强生成", "外部知识库"]
    },
    {
        "query": "Embedding有什么作用？",
        "expected_answer": "Embedding是把文字转换成向量的技术，可以表示文字的语义信息。",
        "expected_claims": ["Embedding", "向量", "语义"]
    },
    {
        "query": "RAG能预测天气吗？",
        "expected_answer": "根据现有资料无法回答该问题。",
        "expected_claims": ["无法回答"]
    },
    {
        "query": "什么是BM25？",
        "expected_answer": "BM25是基于词频和逆文档频率的检索方法。",
        "expected_claims": ["BM25", "词频", "逆文档频率"]
    },
    {
        "query": "RAG和Embedding有什么关系？",
        "expected_answer": "RAG使用Embedding将文档转换为向量，存储在向量数据库中，用于相似度检索。",
        "expected_claims": ["RAG", "Embedding", "向量", "检索"]
    },
    {
        "query": "向量数据库有什么作用？",
        "expected_answer": "向量数据库用于存储和检索向量数据，支持相似度搜索。",
        "expected_claims": ["向量数据库", "存储", "检索", "相似度"]
    },
    {
        "query": "RAG可以防止什么问题？",
        "expected_answer": "RAG可以防止大模型产生幻觉，提供最新信息和私有数据。",
        "expected_claims": ["幻觉", "最新信息", "私有数据"]
    },
    {
        "query": "什么是混合检索？",
        "expected_answer": "混合检索是同时使用向量检索和关键词检索的方法，通过RRF融合排序。",
        "expected_claims": ["混合检索", "向量检索", "关键词检索", "RRF"]
    },
    {
        "query": "RAG系统需要哪些组件？",
        "expected_answer": "RAG系统需要文档加载器、文本分割器、向量数据库、检索器和生成器。",
        "expected_claims": ["文档加载", "文本分割", "向量数据库", "检索器", "生成器"]
    },
    {
        "query": "什么是事实核查？",
        "expected_answer": "事实核查是验证生成回答是否有依据的过程，可以防止幻觉。",
        "expected_claims": ["事实核查", "验证", "依据", "幻觉"]
    }
]
