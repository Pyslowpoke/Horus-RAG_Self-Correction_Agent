import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 获取项目根目录：当前脚本在 src/ 下，往上一级就是项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

def load_and_index_documents(directory_path=None, persist_dir=None):
    if directory_path is None:
        directory_path = str(PROJECT_ROOT / "knowledge_base")
    if persist_dir is None:
        persist_dir = str(PROJECT_ROOT / "chroma_db")

    # 加载 .txt 文件
    txt_loader = DirectoryLoader(
        path=directory_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    txt_docs = txt_loader.load()

    # 加载 .pdf 文件
    pdf_loader = DirectoryLoader(
        path=directory_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    pdf_docs = pdf_loader.load()

    # 合并
    docs = txt_docs + pdf_docs
    print(f"加载了 {len(txt_docs)} 个 txt 文件, {len(pdf_docs)} 个 pdf 文件, 共 {len(docs)} 个文档")

    # 切分文档
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    print(f"切分为 {len(chunks)} 个文档块")

    # 加载 Embedding
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # 计算向量并存入数据库
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print(f"向量库已保存到 {persist_dir}")


if __name__ == "__main__":
    load_and_index_documents()










