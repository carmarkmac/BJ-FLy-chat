from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import sys
import os
import logging
import traceback

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from qa_chain.model_to_llm import model_to_llm
from qa_chain.get_vectordb import get_vectordb

logger = logging.getLogger(__name__)

default_template_rq = """基于以下已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
问题: {question}
已知信息: {context}
"""

class QA_chain_self:
    """"
    不带历史记录的问答链    
    - model：调用的模型名称
    - temperature：温度系数，控制生成的随机性
    - top_k：返回检索的前k个相似文档
    - file_path：建库文件所在路径
    - persist_path：向量数据库持久化路径
    - appid：星火
    - api_key：星火、百度文心、OpenAI、智谱都需要传递的参数
    - Spark_api_secret：星火秘钥
    - Wenxin_secret_key：文心秘钥
    - embeddings：使用的embedding模型  
    - embedding_key：使用的embedding模型的秘钥（智谱或者OpenAI）
    - template：可以自定义提示模板，没有输入则使用默认的提示模板default_template_rq    
    """
    def __init__(self, model:str, temperature:float=0.0, top_k:int=4,  file_path:str=None, persist_path:str=None, appid:str=None, api_key:str=None, Spark_api_secret:str=None,Wenxin_secret_key:str=None, embedding = "openai",  embedding_key = None, template=default_template_rq, vectordb=None):
        logger.info(f"[QA_chain_self.__init__] 初始化开始")
        logger.info(f"[QA_chain_self.__init__] model={model}, embedding={embedding}")
        logger.info(f"[QA_chain_self.__init__] file_path={file_path}")
        logger.info(f"[QA_chain_self.__init__] persist_path={persist_path}")
        logger.info(f"[QA_chain_self.__init__] vectordb 是否预加载: {vectordb is not None}")
        
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.file_path = file_path
        self.persist_path = persist_path
        self.appid = appid
        self.api_key = api_key
        self.Spark_api_secret = Spark_api_secret
        self.Wenxin_secret_key = Wenxin_secret_key
        self.embedding = embedding
        self.embedding_key = embedding_key
        self.template = template
        
        try:
            if vectordb is not None:
                logger.info(f"[QA_chain_self.__init__] 使用预加载的向量数据库")
                self.vectordb = vectordb
            else:
                logger.info(f"[QA_chain_self.__init__] 开始加载向量数据库...")
                self.vectordb = get_vectordb(self.file_path, self.persist_path, self.embedding,self.embedding_key)
            doc_count = self.vectordb._collection.count()
            logger.info(f"[QA_chain_self.__init__] 向量数据库就绪，包含 {doc_count} 个文档")
        except Exception as e:
            logger.error(f"[QA_chain_self.__init__] ❌ 向量数据库加载失败: {e}")
            logger.error(traceback.format_exc())
            raise
        
        try:
            logger.info(f"[QA_chain_self.__init__] 开始初始化 LLM...")
            self.llm = model_to_llm(model=model, temperature=temperature, appid=appid, api_key=api_key, Spark_api_secret=Spark_api_secret,Wenxin_secret_key=Wenxin_secret_key)
            logger.info(f"[QA_chain_self.__init__] LLM 初始化成功")
        except Exception as e:
            logger.error(f"[QA_chain_self.__init__] ❌ LLM 初始化失败: {e}")
            logger.error(traceback.format_exc())
            raise
        
        try:
            logger.info(f"[QA_chain_self.__init__] 开始构建问答链...")
            self.template = template
            QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.vectordb.as_retriever(search_kwargs={"k": top_k}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
            logger.info(f"[QA_chain_self.__init__] 问答链构建成功，初始化完成")
        except Exception as e:
            logger.error(f"[QA_chain_self.__init__] ❌ 问答链构建失败: {e}")
            logger.error(traceback.format_exc())
            raise

    def answer(self, question:str=None, temperature=None, top_k = 4):
        logger.info(f"[QA_chain_self.answer] 收到问题: {question}")
        
        if len(question) == 0:
            return ""
            
        if temperature:
            self.llm.temperature = temperature
        
        logger.info(f"[QA_chain_self.answer] 开始检索并生成答案...")
        result = self.qa_chain({"query": question})
        logger.info(f"[QA_chain_self.answer] 答案生成完成")
        return result["result"]
