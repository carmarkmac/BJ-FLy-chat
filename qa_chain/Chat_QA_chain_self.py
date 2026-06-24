from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
import sys
import os
import logging
import traceback

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from qa_chain.model_to_llm import model_to_llm
from qa_chain.get_vectordb import get_vectordb
import re

logger = logging.getLogger(__name__)

class Chat_QA_chain_self:
    """"
    带历史记录的问答链  
    - model：调用的模型名称
    - temperature：温度系数，控制生成的随机性
    - top_k：返回检索的前k个相似文档
    - chat_history：历史记录，输入一个列表，默认是一个空列表
    - history_len：控制保留的最近 history_len 次对话
    - file_path：建库文件所在路径
    - persist_path：向量数据库持久化路径
    - appid：星火
    - api_key：星火、百度文心、OpenAI、智谱都需要传递的参数
    - Spark_api_secret：星火秘钥
    - Wenxin_secret_key：文心秘钥
    - embeddings：使用的embedding模型
    - embedding_key：使用的embedding模型的秘钥（智谱或者OpenAI）  
    """
    def __init__(self,model:str, temperature:float=0.0, top_k:int=4, chat_history:list=[], file_path:str=None, persist_path:str=None, appid:str=None, api_key:str=None, Spark_api_secret:str=None,Wenxin_secret_key:str=None, embedding = "openai",embedding_key:str=None, vectordb=None):
        logger.info(f"[Chat_QA_chain_self.__init__] 初始化开始")
        logger.info(f"[Chat_QA_chain_self.__init__] model={model}, embedding={embedding}")
        logger.info(f"[Chat_QA_chain_self.__init__] file_path={file_path}")
        logger.info(f"[Chat_QA_chain_self.__init__] persist_path={persist_path}")
        logger.info(f"[Chat_QA_chain_self.__init__] vectordb 是否预加载: {vectordb is not None}")
        
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.chat_history = chat_history
        self.file_path = file_path
        self.persist_path = persist_path
        self.appid = appid
        self.api_key = api_key
        self.Spark_api_secret = Spark_api_secret
        self.Wenxin_secret_key = Wenxin_secret_key
        self.embedding = embedding
        self.embedding_key = embedding_key

        try:
            if vectordb is not None:
                logger.info(f"[Chat_QA_chain_self.__init__] 使用预加载的向量数据库")
                self.vectordb = vectordb
            else:
                logger.info(f"[Chat_QA_chain_self.__init__] 开始加载向量数据库...")
                self.vectordb = get_vectordb(self.file_path, self.persist_path, self.embedding,self.embedding_key)
            doc_count = self.vectordb._collection.count()
            logger.info(f"[Chat_QA_chain_self.__init__] 向量数据库就绪，包含 {doc_count} 个文档")
        except Exception as e:
            logger.error(f"[Chat_QA_chain_self.__init__] ❌ 向量数据库加载失败: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def clear_history(self):
        return self.chat_history.clear()
    
    def change_history_length(self,history_len:int=1):
        n = len(self.chat_history)
        return self.chat_history[n-history_len:]
 
    def answer(self, question:str=None,temperature = None, top_k = 4):
        logger.info(f"[Chat_QA_chain_self.answer] 收到问题: {question}")
        
        if len(question) == 0:
            return "", self.chat_history
        
        if len(question) == 0:
            return ""
            
        if temperature == None:
            temperature = self.temperature
        
        try:
            logger.info(f"[Chat_QA_chain_self.answer] 开始创建 LLM...")
            llm = model_to_llm(self.model, temperature, self.appid, self.api_key, self.Spark_api_secret,self.Wenxin_secret_key)
            logger.info(f"[Chat_QA_chain_self.answer] LLM 创建成功")
        except Exception as e:
            logger.error(f"[Chat_QA_chain_self.answer] ❌ LLM 创建失败: {e}")
            logger.error(traceback.format_exc())
            raise

        # 避免与下方 retriever/chain 重复检索（原先多跑一次 similarity_search，易拖慢与触发超时感）

        try:
            logger.info(f"[Chat_QA_chain_self.answer] 开始创建 retriever (top_k={top_k})...")
            retriever = self.vectordb.as_retriever(search_type="similarity", search_kwargs={'k': top_k})
            logger.info(f"[Chat_QA_chain_self.answer] Retriever 创建成功")
        except Exception as e:
            logger.error(f"[Chat_QA_chain_self.answer] ❌ Retriever 创建失败: {e}")
            logger.error(traceback.format_exc())
            raise

        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""你是北京交通大学非全日制研究生调剂知识库问答助手，请根据下方参考资料，给出详细、完整的回答。

【参考资料】
{context}

【用户问题】
{question}

【回答要求】
1. 基于参考资料中的信息进行回答，不要编造资料中没有的内容
2. 回答要详细、完整，充分展开说明每个要点，不要只做简单概括
3. 对参考资料中的关键信息进行解释和阐述，帮助用户深入理解背景和细节
4. 如果涉及多个方面或步骤，请分点或分段有条理地呈现，逻辑清晰
5. 适当补充对原文信息的理解和说明，使回答更易懂
6. 如果参考资料中没有相关信息，回答：抱歉，知识库中暂无该问题的相关信息

回答：""",
        )

        try:
            logger.info(f"[Chat_QA_chain_self.answer] 开始创建 ConversationalRetrievalChain...")
            qa = ConversationalRetrievalChain.from_llm(
                llm = llm,
                retriever = retriever,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": qa_prompt}
            )
            logger.info(f"[Chat_QA_chain_self.answer] Chain 创建成功，开始执行查询...")
        except Exception as e:
            logger.error(f"[Chat_QA_chain_self.answer] ❌ Chain 创建失败: {e}")
            logger.error(traceback.format_exc())
            raise
        
        try:
            result = qa({"question": question,"chat_history": self.chat_history})
            logger.info(f"[Chat_QA_chain_self.answer] 查询成功")
            docs = result.get("source_documents") or []
            logger.info("[Chat_QA_chain_self.answer] 检索文档数: %d", len(docs))
            if os.environ.get("QA_LOG_RETRIEVAL", "").strip() in ("1", "true", "yes"):
                for i, doc in enumerate(docs[:8]):
                    logger.info("[Chat_QA_chain_self.answer] 文档%d: %s...", i + 1, doc.page_content[:200])
            logger.info(f"[Chat_QA_chain_self.answer] 原始回答: {result.get('answer', '')[:200]}...")
        except Exception as e:
            logger.error(f"[Chat_QA_chain_self.answer] ❌ 查询执行失败: {e}")
            logger.error(traceback.format_exc())
            raise
        
        answer = result['answer']
        self.chat_history.append((question, answer))

        return self.chat_history
