"""加减乘除数学运算"""

import sys
import os
from llama_model import OurLLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
# 从环境变量中读取api_key
api_key = os.getenv('ZISHU_API_KEY')
#print(api_key)
base_url = "https://open.bigmodel.cn/api/paas/v4/"
chat_model = "glm-4-flash"
emb_model = "embedding-3"

llm = OurLLM(api_key=api_key, base_url=base_url, model_name=chat_model)


def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b


def main():

    multiply_tool = FunctionTool.from_defaults(fn=multiply)
    add_tool = FunctionTool.from_defaults(fn=add)

    # 创建ReActAgent实例
    agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)

    response = agent.chat("20+（2*4）等于多少？使用工具计算每一步")

    print(response)


if __name__ == "__main__":
    main()
