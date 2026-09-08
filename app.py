#streamlit run app.py
#终端运行代码↑

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import jieba
import time
import json
import threading
import queue
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from src.generators.llm_client import FaultTolerantLLM
from src.generators.prompt_templates import GENERATION_PROMPT, WEB_SEARCH_PROMPT
from src.retrievers.hyde_retriever import HyDERetriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.web_search_retriever import WebSearchRetriever
from src.verifiers.fact_checker import FactChecker
from src.graph.rag_graph import build_rag_graph
from src.graph.multi_agent_graph import build_multi_agent_rag_graph, NullHybridRetriever
from src.memory.conversation_memory import ConversationMemory
from src.memory.gold_memory_bank import GoldMemoryBank, start_watcher, import_all_corrections

# 用户偏好管理
PREFERENCES_FILE = PROJECT_ROOT / "user_preferences.json"

DEFAULT_PREFERENCES = {
    "nickname": "",
    "disliked_content": "",
    "preferred_content": "",
    "output_format": "简洁回答",
    "age_range": "不想透露",
    "occupation": "不想透露",
    "education_level": "不想透露",
}

def load_preferences():
    prefs = DEFAULT_PREFERENCES.copy()
    if PREFERENCES_FILE.exists():
        try:
            with open(PREFERENCES_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                prefs.update(saved)
        except Exception:
            pass
    return prefs

def save_preferences(prefs):
    with open(PREFERENCES_FILE, "w", encoding="utf-8") as f:
        json.dump(prefs, f, ensure_ascii=False, indent=2)

def get_time_greeting():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "早上好"
    elif 12 <= hour < 14:
        return "中午好"
    elif 14 <= hour < 18:
        return "下午好"
    elif 18 <= hour < 22:
        return "晚上好"
    else:
        return "夜深了"

# 百度 Query 改写（指代消解）
def rewrite_query_with_baidu(query: str) -> str:
    """
    调用百度 Query 改写 API 进行指代消解
    返回改写后的 query，失败时返回原 query
    """
    import requests
    import hashlib

    token = os.getenv("BAIDU_API_KEY")
    if not token:
        return query

    # 缓存检查
    if "rewrite_cache" not in st.session_state:
        st.session_state.rewrite_cache = {}

    cache_key = hashlib.md5(query.encode()).hexdigest()
    if cache_key in st.session_state.rewrite_cache:
        return st.session_state.rewrite_cache[cache_key]

    try:
        url = "https://qianfan.baidubce.com/v2/tools/query_rewrite"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        payload = {"query": query}

        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("code") == 0:
            altered = data.get("altered_query", query)
            st.session_state.rewrite_cache[cache_key] = altered
            return altered
        else:
            return query
    except Exception:
        return query

# 初始化用户偏好
def get_preferences():
    """安全获取用户偏好，确保始终返回有效值"""
    if "preferences" not in st.session_state:
        st.session_state.preferences = load_preferences()
    return st.session_state.preferences

# 自定义 CSS
st.markdown("""
<style>
    h1 { color: #D4A017; text-align: center; font-size: 2.5em; font-weight: 700; }
    h2 { color: #D4A017; font-size: 1.6em; font-weight: 600; }

    hr { border: none; border-top: 2px solid #D4A017; }

    [data-testid="stSidebar"] h1 { color: #D4A017; }

    [data-testid="stAlertContentSuccess"] {
        background-color: #2D7A5F;
        color: #FFFFFF;
    }

    .welcome-icon {
        font-size: 4em;
        text-align: center;
        margin: 20px 0;
    }

    .stButton > button {
        background: #1B4D8C;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        border: none;
    }

    .stButton > button:hover {
        background: #153D6F;
    }
</style>
""", unsafe_allow_html=True)

# 分步缓存初始化
@st.cache_resource
def load_embeddings():
    """加载 Embedding 模型（带预热）"""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    # 预热：强制完成初始化
    _ = embeddings.embed_query("warm up")
    return embeddings

@st.cache_resource
def load_chroma_db(_embeddings):
    return Chroma(
        persist_directory=str(PROJECT_ROOT / "chroma_db"),
        embedding_function=_embeddings
    )

@st.cache_resource
def build_bm25_index(_db):
    all_docs_list = _db.get()["documents"]
    tokenized_docs = [jieba.lcut(doc) for doc in all_docs_list]
    return BM25Okapi(tokenized_docs), all_docs_list

@st.cache_resource
def load_llm():
    primary_config = {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "api_base": "https://api.deepseek.com/v1",
        "model": "deepseek-chat"
    }
    fallback_config = {
        "api_key": os.getenv("SILICONFLOW_API_KEY"),
        "api_base": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen2.5-7B-Instruct"
    }
    return FaultTolerantLLM(primary_config, fallback_config)

@st.cache_resource
def load_light_llm():
    light_config = {
        "api_key": os.getenv("SILICONFLOW_API_KEY"),
        "api_base": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen2.5-7B-Instruct"
    }
    fallback_config = {
        "api_key": os.getenv("SILICONFLOW_API_KEY"),
        "api_base": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen2.5-7B-Instruct"
    }
    return FaultTolerantLLM(light_config, fallback_config)

@st.cache_resource
def load_reranker():
    """加载 Reranker 模型（只加载一次）"""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        model_name = "BAAI/bge-reranker-v2-m3"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()

        return {"tokenizer": tokenizer, "model": model}
    except Exception as e:
        print(f"[Reranker] 加载失败: {e}")
        return None

@st.cache_resource
def load_memory_bank(_embeddings):
    memory_bank = GoldMemoryBank(embeddings=_embeddings)
    import_all_corrections(memory_bank)
    start_watcher(memory_bank)
    return memory_bank

# 延迟初始化（按需加载）
def get_pipeline(search_mode="local"):
    """
    延迟获取流水线组件，根据搜索模式按需加载
    - web 模式：只加载 LLM（秒开）
    - local/hybrid 模式：加载全部组件
    """
    cache_key = f"pipeline_{search_mode}"

    if cache_key in st.session_state:
        return st.session_state[cache_key]

    # web 模式：只加载 LLM + WebSearch
    if search_mode == "web":
        heavy_llm = load_llm()
        light_llm = load_light_llm()

        try:
            web_search_retriever = WebSearchRetriever(max_results=5)
        except ValueError as e:
            st.error(f"❌ {e}")
            st.stop()

        hybrid_retriever = NullHybridRetriever()
        fact_checker = FactChecker(llm_client=light_llm, retriever=None, system_identity=SYSTEM_IDENTITY)

        multi_agent_graph = build_multi_agent_rag_graph(
            heavy_llm=heavy_llm,
            light_llm=light_llm,
            hybrid_retriever=hybrid_retriever,
            web_search_retriever=web_search_retriever,
            fact_checker=fact_checker,
            memory_bank=None,
            hyde_retriever=None,
        )

        result = (None, multi_agent_graph, heavy_llm, fact_checker, web_search_retriever, None)
        st.session_state[cache_key] = result
        return result

    # local/hybrid 模式：加载全部组件
    progress = st.progress(0, text="正在加载 Embedding 模型...")
    embeddings = load_embeddings()
    progress.progress(25, text="正在加载 ChromaDB...")
    
    db = load_chroma_db(embeddings)
    progress.progress(50, text="正在构建 BM25 索引...")
    
    bm25_index, all_docs_list = build_bm25_index(db)
    all_docs = [Document(page_content=doc) for doc in all_docs_list]
    progress.progress(75, text="正在加载 LLM 客户端...")
    
    heavy_llm = load_llm()
    light_llm = load_light_llm()
    memory_bank = load_memory_bank(embeddings)
    reranker_model = load_reranker()  # 加载 Reranker 模型
    progress.progress(100, text="加载完成！")

    hyde_retriever = HyDERetriever(heavy_llm, db)
    hybrid_retriever = HybridRetriever(db, bm25_index, all_docs, reranker_model=reranker_model)
    fact_checker = FactChecker(llm_client=light_llm, retriever=None, system_identity=SYSTEM_IDENTITY)

    try:
        web_search_retriever = WebSearchRetriever(max_results=5)
    except ValueError:
        web_search_retriever = None

    rag_graph = build_rag_graph(hyde_retriever, hybrid_retriever, fact_checker)
    multi_agent_graph = build_multi_agent_rag_graph(
        heavy_llm=heavy_llm,
        light_llm=light_llm,
        hybrid_retriever=hybrid_retriever,
        web_search_retriever=web_search_retriever,
        fact_checker=fact_checker,
        memory_bank=memory_bank,
        hyde_retriever=hyde_retriever,
    )

    result = (rag_graph, multi_agent_graph, heavy_llm, fact_checker, web_search_retriever, memory_bank)
    st.session_state[cache_key] = result
    return result

# 加载系统身份
def load_system_identity():
    try:
        with open(PROJECT_ROOT / "knowledge_base" / "system_profile.txt", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "系统名称：Horus（荷鲁斯）\nSlogan：Insight, not imagination."

SYSTEM_IDENTITY = load_system_identity()

# 页面配置
st.set_page_config(
    page_title="Horus",
    page_icon="𓂀",
    layout="centered",
    initial_sidebar_state="expanded"
)

if "page" not in st.session_state:
    st.session_state.page = "input"
if "trigger_result" not in st.session_state:
    st.session_state.trigger_result = False
if "timeout_counter" not in st.session_state:
    st.session_state.timeout_counter = 0


def get_memory():
    """安全获取对话记忆，确保始终返回有效值"""
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory(max_history=20)
    return st.session_state.memory

# 状态标志驱动跳转
if st.session_state.trigger_result:
    st.session_state.trigger_result = False
    st.session_state.page = "result"
    st.rerun()

# 侧边栏：用户偏好设置
with st.sidebar:
    st.markdown("## ⚙️ 个人设置")

    prefs = get_preferences()

    nickname = st.text_input("你的昵称", value=prefs.get("nickname", ""), placeholder="例如：小明、Tom、小红...")
    disliked = st.text_area(
        "不希望看到的内容（每行一条）",
        value=prefs.get("disliked_content", ""),
        placeholder="例如：\n不要出现英文\n不要技术术语\n不要太长的回答",
        height=100
    )
    preferred = st.text_area(
        "希望看到的内容（每行一条）",
        value=prefs.get("preferred_content", ""),
        placeholder="例如：\n多用中文举例\n给出实际应用场景\n用通俗易懂的语言",
        height=100
    )
    output_format = st.selectbox(
        "输出格式偏好",
        ["简洁回答", "详细回答", "分点列出", "表格形式", "代码示例"],
        index=["简洁回答", "详细回答", "分点列出", "表格形式", "代码示例"].index(prefs.get("output_format", "简洁回答"))
    )

    age_range = st.selectbox(
        "年龄段",
        ["18岁以下", "18-22岁", "23-28岁", "29-35岁", "36-45岁", "45岁以上", "不想透露"],
        index=["18岁以下", "18-22岁", "23-28岁", "29-35岁", "36-45岁", "45岁以上", "不想透露"].index(
            prefs.get("age_range", "不想透露")
        )
    )

    occupation = st.selectbox(
        "职业/身份",
        ["学生", "职场新人（1-3年）", "职场资深（3-10年）", "管理者", "自由职业", "科研/教育", "其他", "不想透露"],
        index=["学生", "职场新人（1-3年）", "职场资深（3-10年）", "管理者", "自由职业", "科研/教育", "其他", "不想透露"].index(
            prefs.get("occupation", "不想透露")
        )
    )

    education_level = st.selectbox(
        "学历",
        ["高中及以下", "本科在读", "本科毕业", "硕士在读", "硕士毕业", "博士及以上", "不想透露"],
        index=["高中及以下", "本科在读", "本科毕业", "硕士在读", "硕士毕业", "博士及以上", "不想透露"].index(
            prefs.get("education_level", "不想透露")
        )
    )

    if st.sidebar.button("💾 保存设置", use_container_width=True):
        new_prefs = {
            "nickname": nickname,
            "disliked_content": disliked,
            "preferred_content": preferred,
            "output_format": output_format,
            "age_range": age_range,
            "occupation": occupation,
            "education_level": education_level,
        }
        save_preferences(new_prefs)
        st.session_state.preferences = new_prefs
        st.sidebar.success("设置已保存！")

    st.markdown("---")
    st.markdown("### 📚 知识库管理")

    if st.sidebar.button("🔄 重建知识库", use_container_width=True):
        with st.spinner("正在重建向量数据库..."):
            try:
                from src.data_ingestion import load_and_index_documents
                load_and_index_documents()
                # 清除缓存，让下次查询用新数据
                st.cache_resource.clear()
                st.sidebar.success("✅ 知识库重建完成！请刷新页面。")
            except Exception as e:
                st.sidebar.error(f"❌ 重建失败：{str(e)}")

    st.markdown("---")
    st.markdown(f"🕐 当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.8em; color:#888; text-align:center; padding:10px;">
        🔒 Horus 不会收集或泄露您的个人信息<br>
        您的输入仅用于更好地回答您的问题
    </div>
    """, unsafe_allow_html=True)

# 页面 1：输入页面
def show_input_page():
    greeting = get_time_greeting()
    prefs = get_preferences()
    nickname = prefs.get("nickname", "")
    greeting_text = f"{greeting}，{nickname}！" if nickname else f"{greeting}！"

    st.markdown('<div class="welcome-icon">𓂀</div>', unsafe_allow_html=True)
    st.markdown("<h1>Horus</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; color:#D4A017; font-size:1.4em; font-weight:600;'>{greeting_text}</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#666; font-size:1.1em;'>Insight, not imagination.</p>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; margin:30px 0;">
        <p style="font-size:1.1em; color:#555;">输入您的问题，Horus 将为您生成答案</p>
    </div>
    """, unsafe_allow_html=True)

    query = st.text_input("", placeholder="请输入您的问题...", label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    search_mode = st.radio(
        "选择搜索模式：",
        ["📚 本地知识库", "🌐 互联网搜索", "🤖 智能混合模式"],
        horizontal=True
    )

    use_web_search = search_mode == "🌐 互联网搜索"
    use_hybrid = search_mode == "🤖 智能混合模式"

    if use_hybrid:
        st.info("🤖 先查本地知识库，内容不足时自动联网补充")
    elif use_web_search:
        st.info("🌐 使用互联网搜索获取最新信息")
    else:
        st.info("📚 使用本地知识库进行检索")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 开始研究", use_container_width=True, type="primary"):
            if query:
                st.session_state.query = query
                st.session_state.use_web_search = use_web_search
                st.session_state.use_hybrid = use_hybrid
                st.session_state.timeout_counter = 0
                st.session_state.page = "processing"
                st.rerun()
            else:
                st.warning("⚠️ 请输入问题")

    # 隐私声明
    st.markdown("""
    <div style="text-align:center; margin-top:40px; padding:15px; background:#f0f4f8; border-radius:10px; border:1px solid #dfe6e9;">
        <p style="font-size:0.9em; color:#636e72; margin:0;">
            🔒 Horus 不会收集或泄露您的个人信息。<br>
            您的输入便于模型基于您的身份和角度理解您的提问，仅用于更好地回答您的问题。<br>
            若不愿填写，也可以在需要时向模型提供这些信息。
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 侧边栏提示
    st.markdown("""
    <div style="text-align:center; margin-top:20px;">
        <p style="font-size:0.9em; color:#888;">
            👉 点击左上角 <strong>⚙️</strong> 打开个人设置，让回答更贴合您的需求
        </p>
    </div>
    """, unsafe_allow_html=True)

# 查询结果缓存
if "query_cache" not in st.session_state:
    st.session_state.query_cache = {}

# 页面 2：研究过程
def show_processing_page():
    original_query = st.session_state.get("query", "")
    use_web_search = st.session_state.get("use_web_search", False)
    use_hybrid = st.session_state.get("use_hybrid", False)

    if use_hybrid:
        initial_mode = "hybrid"
    elif use_web_search:
        initial_mode = "web"
    else:
        initial_mode = "local"

    # ========== 强制日期替换（在进入检索之前） ==========
    from datetime import datetime, timedelta

    def replace_relative_dates(text: str) -> str:
        """把相对时间词替换成绝对日期"""
        now = datetime.now()
        replacements = [
            ("昨天", (now - timedelta(days=1)).strftime("%Y年%m月%d日")),
            ("前天", (now - timedelta(days=2)).strftime("%Y年%m月%d日")),
            ("大前天", (now - timedelta(days=3)).strftime("%Y年%m月%d日")),
            ("明天", (now + timedelta(days=1)).strftime("%Y年%m月%d日")),
            ("后天", (now + timedelta(days=2)).strftime("%Y年%m月%d日")),
            ("大后天", (now + timedelta(days=3)).strftime("%Y年%m月%d日")),
        ]
        for keyword, replacement in replacements:
            if keyword in text:
                text = text.replace(keyword, replacement)
        return text

    # 替换后的 query（搜索引擎永远看不到“昨天”这个词）
    query = replace_relative_dates(original_query)

    # 百度 Query 改写（指代消解）
    rewritten_query = query  # 默认没有改写
    memory = get_memory()
    if len(memory) > 0:
        rewritten = rewrite_query_with_baidu(query)
        if rewritten != query:
            rewritten_query = rewritten  # 保存改写后的查询
            query = rewritten            # 用改写后的查询去搜索

    # 查询结果缓存：相同查询直接返回
    cache_key = f"{query}_{initial_mode}"
    if cache_key in st.session_state.query_cache:
        cached = st.session_state.query_cache[cache_key]
        st.session_state.result = cached["result"]
        memory = get_memory()
        memory.add_user_message(original_query)
        memory.add_assistant_message(cached["result"].get("answer", ""), metadata={"search_mode": initial_mode, "cached": True})
        st.session_state.trigger_result = True
        st.rerun()
        return

    st.markdown(f"<h2>🔍 正在研究：{query}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#666;'>搜索模式：{initial_mode}</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    progress_bar = st.progress(0)
    status_text = st.empty()
    step_times = []

    def show_step(icon, title, desc, progress):
        status_text.markdown(f"{icon} **{title}**\n\n{desc}")
        progress_bar.progress(progress)

    if st.session_state.timeout_counter >= 3:
        st.error("⛔ 连续超时已达 3 次，请切换至互联网搜索模式或简化问题后重试。")
        st.stop()

    t0 = time.time()
    if initial_mode == "web":
        show_step("🤖", "正在初始化...", "加载 LLM 客户端（无需加载 Embedding 模型）", 20)
    else:
        show_step("📥", "正在加载模型...", "首次加载 Embedding 模型（约 25 秒），后续会快很多", 10)

    try:
        _, multi_agent_graph, _, _, _, _ = get_pipeline(search_mode=initial_mode)
    except ValueError as e:
        if "BAIDU_API_KEY" in str(e):
            st.error("❌ 互联网搜索未配置 API Key，请检查 .env 文件中的 BAIDU_API_KEY")
            st.stop()
        raise

    step_times.append(f"初始化: {time.time()-t0:.1f}s")

    t0 = time.time()
    show_step("🔍", "正在优化搜索词...", "指代消解 + 关键词提炼", 40)

    show_step("⚡", "正在执行 Multi-Agent 流水线...", "检索 → 生成 → 核查", 60)

    TIMEOUT_SECONDS = 60

    def run_pipeline():
        current_prefs = get_preferences()
        memory = get_memory()
        chat_history = memory.get_history()

        return multi_agent_graph.invoke({
            "query": query,
            "chat_history": chat_history,
            "search_mode": initial_mode,
            "optimized_query": "",
            "retrieved_docs": [],
            "web_docs": [],
            "context": "",
            "answer": "",
            "verification_log": [],
            "retry_count": 0,
            "memory_context": "",
            "preferences": current_prefs,
        })

    with st.spinner("🤖 **正在执行 Multi-Agent 流水线...**"):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_pipeline)
            try:
                result = future.result(timeout=TIMEOUT_SECONDS)
                st.session_state.timeout_counter = 0
            except FutureTimeoutError:
                future.cancel()
                st.session_state.timeout_counter += 1
                st.session_state.result = {
                    "answer": f"⏱️ 请求处理超时（{TIMEOUT_SECONDS}秒），请尝试简化问题。剩余重试次数：{3 - st.session_state.timeout_counter}",
                    "retrieved_docs": [],
                    "verification_log": [],
                    "retry_count": 0,
                }
                st.session_state.trigger_result = True
                st.rerun()
                return
            except Exception as e:
                st.session_state.result = {
                    "answer": f"处理过程中出错：{str(e)}",
                    "retrieved_docs": [],
                    "verification_log": [],
                    "retry_count": 0,
                }
                st.session_state.trigger_result = True
                st.rerun()
                return

    st.session_state.result = result
    # 记录查询改写信息，供结果页面展示
    st.session_state.query_rewrite_info = {
        "original_query": original_query,
        "rewritten_query": rewritten_query,
        "optimized_query": result.get("optimized_query", ""),
    }
    step_times.append(f"流水线: {time.time()-t0:.1f}s")

    st.session_state.query_cache[cache_key] = {
        "result": result,
        "timestamp": time.time()
    }

    memory = get_memory()
    memory.add_user_message(original_query)
    memory.add_assistant_message(result.get("answer", ""), metadata={"search_mode": initial_mode})

    progress_bar.progress(100)
    total_time = sum(float(t.split(": ")[1].rstrip("s")) for t in step_times)
    status_text.markdown(f"✅ **研究完成！** 总耗时 {total_time:.1f}s（{' → '.join(step_times)}）")

    st.session_state.trigger_result = True
    st.rerun()
# 页面 3：最终结果
def show_result_page():
    result = st.session_state.get("result", {})
    query = st.session_state.get("query", "")
    use_web_search = st.session_state.get("use_web_search", False)
    use_hybrid = st.session_state.get("use_hybrid", False)

    if use_hybrid:
        search_mode = "智能混合"
    elif use_web_search:
        search_mode = "互联网搜索"
    else:
        search_mode = "本地知识库"

    st.markdown(f"<h2>📊 研究结果</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#666;'>问题：{query}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#999;'>搜索模式：{search_mode}</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("<h3>📝 最终回答</h3>", unsafe_allow_html=True)
    answer = result.get("answer", "无结果")
    st.markdown(f"""
    <div class="result-card">
        {answer}
    </div>
    """, unsafe_allow_html=True)

    retry_count = result.get("retry_count", 0)
    if retry_count > 0:
        st.info(f"ℹ️ 经过 {retry_count} 次修正后得出最终答案")
    else:
        st.success("✅ 一次通过，无需修正")

    # 显示查询改写信息（仅互联网搜索模式）
    query_rewrite_info = st.session_state.get("query_rewrite_info", {})
    if query_rewrite_info and (search_mode == "互联网搜索" or search_mode == "智能混合"):
        original = query_rewrite_info.get("original_query", "")
        rewritten = query_rewrite_info.get("rewritten_query", "")
        optimized = query_rewrite_info.get("optimized_query", "")

        # 只展示真正发生变化的查询
        has_rewrite = rewritten != original
        has_optimize = optimized and optimized != rewritten

        if has_rewrite or has_optimize:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<h3>🔍 搜索关键词</h3>", unsafe_allow_html=True)

            # 确定最终用于搜索的词
            final_query = optimized if has_optimize else rewritten

            st.markdown(f"""
            <div style="background:#f0f4f8; padding:15px; border-radius:10px; border-left:4px solid #1B4D8C;">
                <p style="margin:5px 0; color:#636e72;">💬 <strong>你问的：</strong>{original}</p>
                <p style="margin:5px 0; color:#1B4D8C; font-weight:600;">🔍 <strong>实际搜索：</strong>{final_query}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("<h3>🔍 核查日志</h3>", unsafe_allow_html=True)
    verification_log = result.get("verification_log", [])

    if verification_log:
        for i, log in enumerate(verification_log):
            claim = log.get("claim", "")
            verdict = log.get("verdict", "")
            evidence = log.get("evidence", "")

            if verdict == "支持":
                st.markdown(f"""
                <div class="log-success">
                    <strong>🟢 支持</strong><br>
                    <em>{claim}</em>
                    {f'<br><small>证据：{evidence}</small>' if evidence else ''}
                </div>
                """, unsafe_allow_html=True)
            elif verdict == "矛盾":
                st.markdown(f"""
                <div class="log-error">
                    <strong>🔴 矛盾</strong><br>
                    <em>{claim}</em>
                    {f'<br><small>矛盾：{evidence}</small>' if evidence else ''}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="log-warning">
                    <strong>🟡 证据不足</strong><br>
                    <em>{claim}</em>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding:30px; color:#999;">
            <p>📋 无核查日志（回答直接通过）</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<h3>📚 检索到的文档（来源：{search_mode}）</h3>", unsafe_allow_html=True)
    # 兼容本地检索和互联网搜索两种模式
    docs = result.get("retrieved_docs", []) or result.get("web_docs", []) or result.get("all_docs", [])
    if docs:
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "未知")
            url = doc.metadata.get("url", "")
            title = doc.metadata.get("title", f"文档 {i+1}")

            with st.expander(f"📄 {title}", expanded=False):
                if url:
                    st.markdown(f"**来源**: [{source}]({url})")
                else:
                    st.markdown(f"**来源**: {source}")
                st.markdown(f"""
                <div class="step-card">
                    {doc.page_content[:400]}
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3>💬 对话历史</h3>", unsafe_allow_html=True)
    memory = get_memory()
    if len(memory) > 0:
        history = memory.get_history()
        for msg in history[-6:]:
            role_icon = "👤" if msg["role"] == "user" else "🤖"
            role_name = "用户" if msg["role"] == "user" else "助手"
            st.markdown(f"""
            <div style="padding:8px; margin:4px 0; border-radius:8px; background:#f8f9fa;">
                {role_icon} <strong>{role_name}:</strong> {msg["content"][:150]}{"..." if len(msg["content"])>150 else ""}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("暂无对话历史")

    st.markdown("<hr>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 返回重新研究", use_container_width=True):
            st.session_state.page = "input"
            st.rerun()

# 页面路由
if st.session_state.page == "input":
    show_input_page()
elif st.session_state.page == "processing":
    show_processing_page()
elif st.session_state.page == "result":
    show_result_page()
