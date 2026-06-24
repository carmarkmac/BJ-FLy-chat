import sys
import os
import logging
import traceback

sys.path.append("../llm")
from llm.wenxin_llm import Wenxin_LLM
from llm.spark_llm import Spark_LLM
from llm.zhipuai_llm import ZhipuAILLM
from langchain.chat_models import ChatOpenAI
from llm.call_llm import parse_llm_api_key

logger = logging.getLogger(__name__)

def model_to_llm(model:str=None, temperature:float=0.0, appid:str=None, api_key:str=None,Spark_api_secret:str=None,Wenxin_secret_key:str=None):
        """
        星火：model,temperature,appid,api_key,api_secret
        百度问心：model,temperature,api_key,api_secret
        智谱：model,temperature,api_key
        OpenAI：model,temperature,api_key
        """
        logger.info(f"[model_to_llm] 开始创建 LLM: model={model}, temp={temperature}")
        
        if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k", "deepseek-chat", "deepseek-reasoner"]:
            if api_key == None:
                logger.info(f"[model_to_llm] 从环境变量获取 OPENAI_API_KEY")
                api_key = parse_llm_api_key("openai")
            openai_api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
            logger.info(f"[model_to_llm] 使用 OpenAI 兼容接口")
            logger.info(f"[model_to_llm] model_name={model}")
            logger.info(f"[model_to_llm] openai_api_base={openai_api_base}")
            logger.info(f"[model_to_llm] api_key={'已设置' if api_key else '未设置'}")
            
            try:
                # 知识库问答常含长上下文，DeepSeek 等接口偶发较慢；默认放宽读超时（秒）
                try:
                    _t = float(os.environ.get("LLM_HTTP_TIMEOUT_SEC", "600"))
                except ValueError:
                    _t = 600.0
                try:
                    _retries = int(os.environ.get("LLM_MAX_RETRIES", "2"))
                except ValueError:
                    _retries = 2
                llm = ChatOpenAI(
                    model_name=model,
                    temperature=temperature,
                    openai_api_key=api_key,
                    openai_api_base=openai_api_base,
                    request_timeout=_t,
                    max_retries=_retries,
                )
                logger.info(
                    "[model_to_llm] ChatOpenAI 已创建 request_timeout=%s max_retries=%s",
                    _t,
                    _retries,
                )
                return llm
            except Exception as e:
                logger.error(f"[model_to_llm] ❌ ChatOpenAI 创建失败: {e}")
                logger.error(traceback.format_exc())
                raise
        elif model in ["ERNIE-Bot", "ERNIE-Bot-4", "ERNIE-Bot-turbo"]:
            if api_key == None or Wenxin_secret_key == None:
                api_key, Wenxin_secret_key = parse_llm_api_key("wenxin")
            logger.info(f"[model_to_llm] 使用文心 API")
            llm = Wenxin_LLM(model=model, temperature = temperature, api_key=api_key, secret_key=Wenxin_secret_key)
            return llm
        elif model in ["Spark-1.5", "Spark-2.0"]:
            if api_key == None or appid == None and Spark_api_secret == None:
                api_key, appid, Spark_api_secret = parse_llm_api_key("spark")
            logger.info(f"[model_to_llm] 使用星火 API")
            llm = Spark_LLM(model=model, temperature = temperature, appid=appid, api_secret=Spark_api_secret, api_key=api_key)
            return llm
        elif model in ["chatglm_pro", "chatglm_std", "chatglm_lite"]:
            if api_key == None:
                api_key = parse_llm_api_key("zhipuai")
            logger.info(f"[model_to_llm] 使用智谱 API")
            llm = ZhipuAILLM(model=model, zhipuai_api_key=api_key, temperature = temperature)
            return llm
        else:
            error_msg = f"model {model} not support!!!"
            logger.error(f"[model_to_llm] ❌ {error_msg}")
            raise ValueError(error_msg)
