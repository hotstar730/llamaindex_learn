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
from typing import List
from sqlalchemy import insert
from IPython.display import Markdown, display
from llama_index.core import SQLDatabase
from llama_index.core.indices.struct_store import NLSQLTableQueryEngine
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer, text,
)

from llm_service.agents.agent_base import AgentBase
from llm_service.util.excel_util import ExcelUtil


class AgentSql(AgentBase):

    def __init__(self) -> None:
        filepath = './data/excel/test.xlsx'
        print("init")
        filepath = './data/excel/test.xlsx'
        table_name = os.path.splitext(os.path.basename(filepath))[0]

        self.meta_data = MetaData()
        schema = Table(
            table_name,
            self.meta_data,
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
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        self.engine = create_engine("sqlite:///:memory:")
        engine = create_engine("mysql+mysqldb://scott:tiger@hostname/dbname", pool_recycle=3600, echo=True)
        self.meta_data.create_all(self.engine)
        self.database = SQLDatabase(self.engine, include_tables=[table_name])

        # 插入数据
        rows = self._read_excel(filepath)
        self.connect = self.database.engine.connect()
        for row in rows:
            stmt = insert(schema).values(**row)
            # with self.database.engine.begin() as connection:
            cursor = self.connect.execute(stmt)

        # 定义你的LLM
        self.llm = Ollama(model="pxlksr/defog_sqlcoder-7b-2:Q8")
        self.llm.temperature = 0.2
        self.llm.base_url = "http://1.92.64.112:11434"

        # 查询引擎
        self.query_engine = NLSQLTableQueryEngine(
            sql_database=self.database, tables=[table_name], llm=self.llm
        )

        print('init finish')


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
        # self.metadata_obj.create_all(self.sql_database.engine)
        rows = self.connect.execute(text("SELECT vin from test"))
        for row in rows:
            print(row)

        if messages:
            message = messages[-1]
            response = self.query_engine.query(message.content + "")
            return ChatMessage(role="assistant", content=response)
        else:
            return ChatMessage(role="assistant", content="can i help you!")