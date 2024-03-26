#!/usr/bin/python3
# encoding=utf-8

"""
    项目名称:  llamaindex_learn
    文件名称： 03_sql_simple.py
    功能描述： ..
    创建者：   lixinxin
    创建日期： 2024/3/14 13:47
    需要使用mysql数据进行测试
"""

from typing import List

from llama_index.core import SQLDatabase
from llama_index.core.indices.struct_store import NLSQLTableQueryEngine
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
from sqlalchemy import (
    create_engine,
    MetaData,
    text,
)
from llm_service.util.excel_util import ExcelUtil
from llm_service.agents.agent_base import AgentBase


class AgentSql(AgentBase):

    def __init__(self) -> None:
        # 需要加载了表
        table_name = ['盘点结果']

        self.meta_data = MetaData()
        self.engine = create_engine("mysql://root:Foton12345&@1.92.64.112/llama", pool_recycle=3306, echo=True)
        self.meta_data.create_all(self.engine)
        self.database = SQLDatabase(self.engine, include_tables=table_name)

        # 读取成功
        self.connect = self.database.engine.connect()
        rows = self.connect.execute(text("SELECT vin from 盘点结果 limit 10"))
        for row in rows:
            print(row)

        # 定义你的LLM
        self.llm = Ollama(model="pxlksr/defog_sqlcoder-7b-2:Q8")
        self.llm.temperature = 0.2
        self.llm.base_url = "http://1.92.64.112:11434"

        # 查询引擎
        self.query_engine = NLSQLTableQueryEngine(
            sql_database=self.database, tables=table_name, llm=self.llm
        )

    def _read_excel(self, filename):
        data_rows = []
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
            data_rows.append(map)

        return data_rows

    def chat(self, messages: List[ChatMessage]) -> ChatMessage:
        if messages:
            message = messages[-1]
            self.query_engine.update_prompts('use mysql database')
            response = self.query_engine.query(message.content + "")
            return ChatMessage(role="assistant", content=response)
        else:
            return ChatMessage(role="assistant", content="can i help you!")