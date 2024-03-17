#!/usr/bin/python3
# encoding=utf-8

"""
    项目名称:  llamaindex_learn
    文件名称： 03_sql_simple.py
    功能描述： ..
    创建者：   lixinxin
    创建日期： 2024/3/14 13:47
"""
import os

import openai
from llama_index.core import SQLDatabase
from llama_index.llms.ollama import Ollama
from llm_service.util.excel_util import ExcelUtil
import os
from typing import List
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama

from llm_service.agents.agent_base import AgentBase
from IPython.display import Markdown, display

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
)

os.environ["OPENAI_API_KEY"] = "sk-.."
openai.api_key = os.environ["OPENAI_API_KEY"]

#
# class AgentDocument(AgentBase):
#     def __init__(self) -> None:
#         # 没有用到，不填会报错
#         os.environ["OPENAI_API_KEY"] = "sk-.."
#         openai.api_key = os.environ["OPENAI_API_KEY"]
#
#         engine = create_engine("sqlite:///:memory:")
#         metadata_obj = MetaData()
#
#         sheet = 'Sheet1'
#         filepath = '../data/excel/test.xlsx'
#         table_name = os.path.splitext(os.path.basename(filepath))[0]
#         col_name, rows = read_excel(filepath, sheet)
#
#         city_stats_table = Table(
#             table_name,
#             metadata_obj,
#             Column("vin", String(16), primary_key=True),
#             Column("库存类型", String(16), nullable=False),
#             Column("是否上牌", String(16), nullable=False),
#             Column("经销商编码", String(16), nullable=False),
#             Column("经销商名称", String(16), nullable=False),
#             Column("送达方", String(16)),
#             Column("终端离线时长", Integer),
#             Column("车联网实销状态", String(16), nullable=False),
#             Column("TSP库存名称", String(16)),
#             Column("DMS库存类型", String(16)),
#             Column("SAP库存类型", String(16)),
#             Column("车联网库存类型与（DMS库存类型是否匹配或者SAP库存类型）", String(16), nullable=False),
#             Column("不一致原因", String(16)),
#             Column("入中心库时间", String(16)),
#             Column("库龄", String(16)),
#         )
#         metadata_obj.create_all(engine)
#
#         sql_database = SQLDatabase(engine, include_tables=[table_name])
#         from sqlalchemy import insert
#
#         for row in rows:
#             stmt = insert(city_stats_table).values(**row)
#             with engine.begin() as connection:
#                 cursor = connection.execute(stmt)
#
#         # 定义你的LLM
#         llm = Ollama(model="pxlksr/defog_sqlcoder-7b-2:Q8")
#         llm.temperature = 0.2
#         llm.base_url = "http://1.92.64.112:11434"
#
#         query_engine = NLSQLTableQueryEngine(
#             sql_database=sql_database, tables=[table_name], llm=llm
#         )
#
#         Settings.llm = Ollama(model="qwen:7b-chat")
#         Settings.llm.base_url = "http://1.92.64.112:11434"
#         # embed_model
#         Settings.embed_model = resolve_embed_model("local:data/embed_model/bge-small-en-v1.5")
#         # 文档解析成更小的块
#         # Settings.chunk_size = 512
#         documents = SimpleDirectoryReader("./data/document").load_data()
#         self.index = VectorStoreIndex.from_documents(documents)
#         self.query_engine = self.index.as_chat_engine()
#
#     def _read_excel(filename, sheet):
#         res = []
#         ex = ExcelUtil(filename)
#         max_row = ex.get_max_row()
#         head = []
#         for j in range(max_row):
#             i = j + 1
#             if i == 1:
#                 head = ex.get_row_value(i)
#                 continue
#
#             map = {}
#             row = ex.get_row_value(i)
#             for r in range(len(row)):
#                 map[head[r]] = row[r]
#             print(i, " ", row)
#             res.append(map)
#
#         return head, res
#
#     def chat(self, messages: List[ChatMessage]) -> ChatMessage:
#         query_str = "终端离线时长小于10的经销商名称分别是哪些"
#         response = query_engine.query(query_str)
#         print(response)
#         display(Markdown(f"<b>{response}</b>"))
#
#         if messages:
#             message = messages[-1]
#             response = self.query_engine.chat(message.content + "")
#             return ChatMessage(role="assistant", content=response)
#         else:
#             return ChatMessage(role="assistant", content="can i help you!")


def read_excel(filename, sheet):
    res = []
    ex = ExcelUtil(filename)
    max_row = ex.get_max_row()
    head = []
    for j in range(max_row):
        i = j + 1
        if i == 1:
            head = ex.get_row_value(i)
            continue

        map = {}
        row = ex.get_row_value(i)
        for r in range(len(row)):
            map[head[r]] = row[r]
        print(i, " ", row)
        res.append(map)

    return head, res

if __name__ == "__main__":
    engine = create_engine("sqlite:///:memory:")
    metadata_obj = MetaData()

    sheet = 'Sheet1'
    filepath = '../data/excel/test.xlsx'
    table_name = os.path.splitext(os.path.basename(filepath))[0]
    col_name, rows = read_excel(filepath, sheet)

    city_stats_table = Table(
        table_name,
        metadata_obj,
        Column("vin", String(16), primary_key=True),
        Column("库存类型", String(16), nullable=False),
        Column("是否上牌", String(16), nullable=False),
        Column("经销商编码", String(16), nullable=False),
        Column("经销商名称", String(16), nullable=False),
        Column("送达方", String(16)),
        Column("终端离线时长", Integer),
        Column("车联网实销状态", String(16), nullable=False),
        Column("TSP库存名称", String(16)),
        Column("DMS库存类型", String(16)),
        Column("SAP库存类型", String(16)),
        Column("车联网库存类型与（DMS库存类型是否匹配或者SAP库存类型）", String(16), nullable=False),
        Column("不一致原因", String(16)),
        Column("入中心库时间", String(16)),
        Column("库龄", String(16)),
    )
    metadata_obj.create_all(engine)

    sql_database = SQLDatabase(engine, include_tables=[table_name])
    from sqlalchemy import insert

    for row in rows:
        stmt = insert(city_stats_table).values(**row)
        with engine.begin() as connection:
            cursor = connection.execute(stmt)

    from sqlalchemy import text

    with engine.connect() as con:
        rows = con.execute(text("SELECT vin from " + table_name))
        for row in rows:
            print(row)

    from llama_index.core.query_engine import NLSQLTableQueryEngine

    # 定义你的LLM
    llm = Ollama(model="pxlksr/defog_sqlcoder-7b-2:Q8")
    llm.temperature = 0.2
    llm.base_url = "http://1.92.64.112:11434"

    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database, tables=[table_name], llm=llm
    )
    query_str = "终端离线时长小于10的经销商名称分别是哪些"
    response = query_engine.query(query_str)
    print(response)
    display(Markdown(f"<b>{response}</b>"))

