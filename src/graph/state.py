#定义流水线传递哪些数据
from typing import TypedDict,List,Any
from langchain_core.documents import Document

class RAGState(TypedDict):
    query:str
    retrieved_docs:List[Document]
    context:str
    answer:str
    verification_log:List[Any]
    retry_count:int
