import os
from dotenv import load_dotenv
from llama_model import OurLLM

# 加载环境变量
load_dotenv()
# 从环境变量中读取api_key
api_key = os.getenv('ZISHU_API_KEY')
bocha_api_key = os.getenv('BOCHA_API_KEY')
base_url = "https://open.bigmodel.cn/api/paas/v4/"
chat_model = "glm-4-flash"
emb_model = "embedding-3"


llm = OurLLM(api_key=api_key, base_url=base_url, model_name=chat_model)
# response = llm.stream_complete("你是谁？")
# for chunk in response:
#     print(chunk, end="", flush=True)



# 搜索功能搭建
from llama_index.core.tools import FunctionTool
import requests

# 需要先把BOCHA_API_KEY填写到.env文件中去。
BOCHA_API_KEY = os.getenv('BOCHA_API_KEY')


# 定义Bocha Web Search工具
def bocha_web_search_tool(query: str, count: int = 8) -> str:
    """
    使用Bocha Web Search API进行联网搜索，返回搜索结果的字符串。

    参数:
    - query: 搜索关键词
    - count: 返回的搜索结果数量

    返回:
    - 搜索结果的字符串形式
    """
    url = 'https://api.bochaai.com/v1/web-search'
    headers = {
        'Authorization': f'Bearer {BOCHA_API_KEY}',  # 请替换为你的API密钥
        'Content-Type': 'application/json'
    }
    data = {
        "query": query,
        "freshness": "noLimit",  # 搜索的时间范围，例如 "oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit"
        "summary": True,  # 是否返回长文本摘要总结
        "count": count
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        # 返回给大模型的格式化的搜索结果文本
        # 可以自己对博查的搜索结果进行自定义处理
        return str(response.json())
    else:
        raise Exception(f"API请求失败，状态码: {response.status_code}, 错误信息: {response.text}")


search_tool = FunctionTool.from_defaults(fn=bocha_web_search_tool)
from llama_index.core.agent import ReActAgent

agent = ReActAgent.from_tools([search_tool], llm=llm, verbose=True)

# 测试用例
query = "2024票房最高的电影是？"
response = agent.chat(f"请帮我搜索以下内容：{query}")
print('结果：',response)