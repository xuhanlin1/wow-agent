import sqlite3

# 创建数据库(已运行创建过）
# sqllite_path = 'llmdb.db'
# con = sqlite3.connect(sqllite_path)
#
# # 创建表
# sql = """
# CREATE TABLE `section_stats` (
#   `部门` varchar(100) DEFAULT NULL,
#   `人数` int(11) DEFAULT NULL
# );
# """
# c = con.cursor()
# cursor = c.execute(sql)
# c.close()
# con.close()

# 然后给数据库填充一些数据（已运行）
# con = sqlite3.connect(sqllite_path)
# c = con.cursor()
# data = [
#     ["专利部",22],
#     ["商标部",25],
# ]
# for item in data:
#     sql = """
#     INSERT INTO section_stats (部门,人数)
#     values('%s','%d')
#     """%(item[0],item[1])
#     c.execute(sql)
#     con.commit()
# c.close()
# con.close()

from llama_model import OurLLM
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.tools import QueryEngineTool
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import create_engine, select
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()
# 从环境变量中读取api_key
api_key = os.getenv('ZISHU_API_KEY')
base_url = "https://open.bigmodel.cn/api/paas/v4/"
chat_model = "glm-4-flash"
llm = OurLLM(api_key=api_key, base_url=base_url, model_name=chat_model)

# 配置本地大模型
Settings.llm = llm

## 创建数据库查询引擎
engine = create_engine("sqlite:///llmdb.db")
# prepare data
sql_database = SQLDatabase(engine, include_tables=["section_stats"])
query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["section_stats"],
    llm=Settings.llm,
    embed_model="local"

)


# 创建工具函数
def multiply(a: float, b: float) -> float:
    """将两个数字相乘并返回乘积。"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: float, b: float) -> float:
    """将两个数字相加并返回它们的和。"""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)

# 把数据库查询引擎封装到工具函数对象中
staff_tool = QueryEngineTool.from_defaults(
    query_engine,
    name="section_staff",
    description="查询部门的人数。"
)


# 构建ReActAgent，可以加很多函数，在这里只加了加法函数和部门人数查询函数。
agent = ReActAgent.from_tools([add_tool, staff_tool], verbose=True)
# 通过agent给出指令
response = agent.chat("请从数据库表中获取`专利部`和`商标部`的人数，并将这两个部门的人数相加！")