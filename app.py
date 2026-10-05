#streamlit run app.py
#终端运行代码↑

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import streamlit as st
import time
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from src.retrievers.hyde_retriever import HyDERetriever
from src.retrievers.web_search_retriever import WebSearchRetriever
from src.verifiers.fact_checker import FactChecker
from src.graph.multi_agent_graph import build_multi_agent_rag_graph
from src.memory.conversation_memory import ConversationMemory
from src.memory.gold_memory_bank import GoldMemoryBank, start_watcher, import_all_corrections

from src.config import load_config, index_version, fingerprint, embedding_signature
from src import components
from src.runtime import RequestBudget, RequestTimeout, request_scope, timed

CONFIG = load_config()
import logging
logging.basicConfig(level=getattr(logging, CONFIG["logging"]["level"], logging.INFO))

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

# Models are cached across requests; index-dependent resources use the manifest version.
@st.cache_resource
def load_embeddings():
    return components.load_embeddings(CONFIG)

@st.cache_resource
def load_chroma_db(_embeddings, version):
    return components.load_db(CONFIG, _embeddings)

@st.cache_resource
def build_bm25_index(_db, version):
    return components.build_bm25(_db)

@st.cache_resource
def load_llm():
    return components.make_llm(CONFIG)

@st.cache_resource
def load_light_llm():
    return components.make_llm(CONFIG, light=True)

@st.cache_resource
def load_reranker():
    return components.load_reranker(CONFIG)

@st.cache_resource
def load_memory_bank(_embeddings):
    # Separate memory vectors when changing the embedding model.
    signature = fingerprint(embedding_signature(CONFIG))[:12]
    legacy = (CONFIG['embedding']['model_name'] == 'all-MiniLM-L6-v2'
              and not CONFIG['embedding']['normalize_embeddings']
              and not CONFIG['embedding']['query_prefix'])
    memory_bank = GoldMemoryBank(embeddings=_embeddings,
        persist_dir=str(PROJECT_ROOT / 'gold_memory_db' if legacy else PROJECT_ROOT / 'gold_memory_db' / signature),
        distance_threshold=CONFIG.get('memory', {}).get('distance_threshold', 0.65))
    import_all_corrections(memory_bank)
    start_watcher(memory_bank)
    return memory_bank

@st.cache_resource
def get_executor():
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix='rag')

@st.cache_resource
def get_pipeline(search_mode='local', version='', verification_enabled=True):
    heavy_llm, light_llm = load_llm(), load_light_llm()
    web = None
    if search_mode in ('web', 'hybrid'):
        try:
            web = WebSearchRetriever(max_results=CONFIG['web']['max_results'], timeout=CONFIG['web']['timeout'])
        except ValueError:
            if search_mode == 'web':
                raise
    retriever, memory_bank, hyde = None, None, None
    if search_mode not in ('web', 'self_aware'):
        with timed('load_embeddings'):
            embeddings = load_embeddings()
        with timed('load_index'):
            db = load_chroma_db(embeddings, version)
            bm25, docs = build_bm25_index(db, version)
        with timed('load_reranker'):
            reranker = load_reranker()
        retriever = components.make_retriever(CONFIG, db, bm25, docs, reranker)
        with timed('load_memory'):
            memory_bank = load_memory_bank(embeddings)
        if CONFIG['retrieval']['hyde_enabled']:
            hyde = HyDERetriever(light_llm, db, max_tokens=CONFIG['retrieval']['hyde_max_tokens'])
    checker = FactChecker(light_llm, max_retries=CONFIG['generation']['max_retries'])
    graph = build_multi_agent_rag_graph(heavy_llm, light_llm, retriever, web, checker, memory_bank,
        hyde_retriever=hyde, max_retries=CONFIG['generation']['max_retries'],
        verification_enabled=verification_enabled,
        context_max_chars=CONFIG['generation']['context_max_chars'], top_k=CONFIG['retrieval']['top_k'],
        streaming=CONFIG['generation']['streaming'], web_min_lexical_score=CONFIG['retrieval']['min_lexical_score'])
    return graph, bool(retriever and retriever.reranker_model)

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

    if st.sidebar.button("💾 保存设置", width="stretch"):
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

    if st.sidebar.button("🔄 重建知识库", width="stretch"):
        with st.spinner("正在重建向量数据库..."):
            try:
                from src.data_ingestion import load_and_index_documents
                load_and_index_documents()
                # 清除缓存，让下次查询用新数据
                st.session_state.query_cache = {}
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

    query = st.text_input("问题", placeholder="请输入您的问题...", label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    search_mode = st.radio(
        "选择搜索模式：",
        ["📚 本地知识库", "🌐 互联网搜索", "🤖 智能混合模式"],
        horizontal=True
    )
    verify_request = st.checkbox('回答后执行证据核查（耗时更长）', value=False,
        help='默认先检索并生成带引用的回答，不代表已核查。需要逐条核验时开启；核查失败会保留已有回答。')

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
        if st.button("🚀 开始研究", width="stretch", type="primary"):
            if query:
                st.session_state.query = query
                st.session_state.use_web_search = use_web_search
                st.session_state.use_hybrid = use_hybrid
                st.session_state.verify_request = verify_request
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
    original_query = st.session_state.get('query', '')
    mode = 'hybrid' if st.session_state.get('use_hybrid') else ('web' if st.session_state.get('use_web_search') else 'local')
    memory = get_memory()
    history = memory.get_history()
    preferences = dict(get_preferences())
    verification_enabled = bool(st.session_state.get('verify_request', False) and CONFIG['generation']['verification_enabled'])
    request_config = {**CONFIG, 'generation': {**CONFIG['generation'], 'verification_enabled': verification_enabled}}
    version = index_version(CONFIG)
    from src.query import replace_relative_dates, contextual_query, cache_key_for
    query = replace_relative_dates(original_query)
    search_query = contextual_query(query, history)
    key = cache_key_for(query, mode, history, preferences, version, request_config)
    cache = st.session_state.query_cache
    cached = cache.get(key)
    if cached and time.time() - cached['timestamp'] < CONFIG['runtime']['query_cache_ttl']:
        result = cached['result']
    else:
        previous = st.session_state.get('active_future')
        if previous is not None and not previous.done():
            st.info('上一个请求正在停止，请稍后重试。')
            if st.button('检查任务状态'):
                st.rerun()
            return
        budget = RequestBudget(CONFIG['runtime']['request_timeout'])
        state = {'query': query, 'optimized_query': search_query, 'search_mode': mode,
                 'chat_history': history, 'preferences': preferences,
                 'enhanced_context': '最近对话（仅用于理解指代，不作为事实依据）：\n' + str(history[-4:])[:2000],
                 'top_k': CONFIG['retrieval']['top_k'], 'retry_count': 0,
                 'verification_status': 'pending' if verification_enabled else 'disabled'}

        def run_pipeline():
            with request_scope(budget):
                with timed('initialization'):
                    from src.agents.router_agent import router_agent
                    effective_mode = router_agent(state).get('search_mode', mode)
                    graph, reranker_active = get_pipeline(effective_mode, version, verification_enabled)
                if CONFIG['web']['query_rewrite_enabled']:
                    from src.query import rewrite_query
                    state['optimized_query'] = rewrite_query(search_query, CONFIG['web']['timeout'])
                output = graph.invoke(state, config={'recursion_limit': 12 + 2 * CONFIG['generation']['max_retries']})
                output['metrics'] = budget.metrics
                output['metrics']['total_seconds'] = time.monotonic() - budget.started
                output['reranker_active'] = reranker_active
                return output

        future = get_executor().submit(run_pipeline)
        st.session_state.active_future = future
        stage = st.empty()
        draft = st.empty()
        draft_text = ''
        stage_labels = {'initialization': '正在加载模型', 'node.retrieval': '正在查找相关资料',
                        'node.generation': '正在组织回答', 'node.fact_check': '正在核对证据',
                        'node.rewrite': '正在修正回答', 'node.web_search': '正在搜索互联网'}
        with st.spinner('正在处理…'):
            try:
                while True:
                    while not budget.events.empty():
                        kind, value = budget.events.get_nowait()
                        if kind == 'stage' and value in stage_labels:
                            stage.info(stage_labels[value])
                        elif kind == 'draft_reset':
                            draft_text = ''
                            draft.empty()
                        elif kind == 'token':
                            draft_text += value
                            draft.markdown('**回答草稿（完成证据核查后显示最终结果）**\n\n' + draft_text)
                    try:
                        result = future.result(timeout=min(0.15, budget.remaining()))
                        break
                    except FutureTimeoutError:
                        if future.done():
                            raise

                st.session_state.timeout_counter = 0
            except (FutureTimeoutError, RequestTimeout):
                budget.cancelled.set()
                future.cancel()
                st.session_state.timeout_counter += 1
                result = {**budget.last_result,
                          'answer': budget.last_result.get('answer', '请求超时，请稍后重试。首次模型加载可能需要更长时间。'),
                          'verification_status': 'timeout', 'error': True, 'metrics': budget.metrics}
            except Exception:
                logging.exception('RAG 请求失败')
                result = {**budget.last_result,
                          'answer': budget.last_result.get('answer', '请求处理失败，请检查模型、索引及 API 配置。'),
                          'verification_status': 'error', 'error': True, 'metrics': budget.metrics}
        # Cache only successful, fully evaluated results, with a bounded lifetime/size.
        if not result.get('error') and not any(result.get(k) for k in ('generation_error', 'retrieval_error', 'web_error')) and result.get('verification_status') not in ('error', 'failed', 'pending'):
            cache[key] = {'result': result, 'timestamp': time.time()}
            while len(cache) > CONFIG['runtime']['query_cache_max_entries']:
                del cache[next(iter(cache))]
    st.session_state.result = result
    st.session_state.query_rewrite_info = {'original_query': original_query,
        'rewritten_query': query, 'optimized_query': result.get('optimized_query', search_query)}
    memory.add_user_message(original_query)
    memory.add_assistant_message(result.get('answer', ''), metadata={'search_mode': mode})
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

    status = result.get('verification_status', 'skipped')
    labels = {'passed': '核查通过', 'failed': '仍有陈述缺少证据，请结合下方核查结果阅读',
              'error': '核查或请求失败，本次结果未验证', 'disabled': '本次未启用核查',
              'skipped': '本次未执行事实核查', 'no_claims': '未提取到可核查陈述',
              'timeout': '请求已超时；如已有回答，已保留，但核查未完成'}
    if status == 'passed':
        st.success(labels[status])
    else:
        st.info(labels.get(status, '核查状态未知'))
    if result.get('retry_count'):
        st.caption(f"已修正 {result['retry_count']} 次")
    if not result.get('reranker_active') and search_mode != '互联网搜索' and result.get('search_mode') != 'self_aware':
        st.caption('本地检索使用词项过滤；重排序未启用或加载失败。')
    if result.get('metrics'):
        with st.expander('耗时与调用统计'):
            st.json(result['metrics'])

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
            <p>📋 无核查日志（不代表核查通过）</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<h3>📚 检索到的文档（来源：{search_mode}）</h3>", unsafe_allow_html=True)
    # 兼容本地检索和互联网搜索两种模式
    docs = result.get("all_docs", [])
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
        if st.button("🔄 返回重新研究", width="stretch"):
            st.session_state.page = "input"
            st.rerun()

# 页面路由
if st.session_state.page == "input":
    show_input_page()
elif st.session_state.page == "processing":
    show_processing_page()
elif st.session_state.page == "result":
    show_result_page()
