import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
# 从环境变量中读取api_key
api_key = os.getenv('ZISHU_API_KEY')
base_url = "https://open.bigmodel.cn/api/paas/v4/"
chat_model = "glm-4-flash"
emb_model = "embedding-3"

# 配置对话模型
from llama_index.llms.zhipuai import ZhipuAI
llm = ZhipuAI(
    api_key = api_key,
    model = chat_model,
)

# 配置嵌入模型
from llama_index.embeddings.zhipuai import ZhipuAIEmbedding
embedding = ZhipuAIEmbedding(
    api_key = api_key,
    model = emb_model,
)


# 构建索引
# 从指定文件读取，输入为List
from llama_index.core import SimpleDirectoryReader,Document
documents = SimpleDirectoryReader(input_files=['./docs/新闻1.txt']).load_data()

# 构建节点
from llama_index.core.node_parser import SentenceSplitter
transformations = [SentenceSplitter(chunk_size = 512)]

from llama_index.core.ingestion.pipeline import run_transformations
nodes = run_transformations(documents, transformations=transformations)

# 构建索引
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from llama_index.core import StorageContext, VectorStoreIndex

emb = embedding.get_text_embedding("你好呀呀")  ## 这边的字符串应该随意，主要是为了获得embedding的长度
print(len(emb))
vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(len(emb)))
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    nodes = nodes,
    storage_context=storage_context,
    embed_model = embedding,
)

# 构建问答引擎
# 构建检索器
from llama_index.core.retrievers import VectorIndexRetriever
# 想要自定义参数，可以构造参数字典
kwargs = {'similarity_top_k': 5, 'index': index, 'dimensions': len(emb)} # 必要参数
retriever = VectorIndexRetriever(**kwargs)

# 构建合成器
from llama_index.core.response_synthesizers  import get_response_synthesizer
response_synthesizer = get_response_synthesizer(llm=llm, streaming=True)

# 构建问答引擎
from llama_index.core.query_engine import RetrieverQueryEngine
engine = RetrieverQueryEngine(
      retriever=retriever,
      response_synthesizer=response_synthesizer,
        )

# 提问
question = "全国铁路旅客发送量?"
response = engine.query(question)
for text in response.response_gen:
    print(text, end="")