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

os.environ["OPENAI_API_KEY"] = "sk-.."
openai.api_key = os.environ["OPENAI_API_KEY"]

from IPython.display import Markdown, display

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
)

from datetime import date
from xlrd import open_workbook, xldate_as_tuple
import os

def read_excel(path, file_name, sheet):
    res = []
    with open_workbook(os.path.join(path, file_name+'.xlsx')) as workbook:
        worksheet = workbook.sheet_by_name(sheet)
        for row_index in range(0, worksheet.nrows):
            # 获取列名
            if row_index == 0:
                col_name = [worksheet.cell_value(0, i) for i in range(worksheet.ncols)]
                continue
            map = {}
            for col_index in range(worksheet.ncols):
                # 判断单元格里的值是否是日期
                if worksheet.cell_type(row_index, col_index) == 3:
                    # 先将单元格里的表示日期数值转换成元组
                    date_cell = xldate_as_tuple(worksheet.cell_value(row_index, col_index), workbook.datemode)
                    # 使用元组的索引来引用元组的前三个元素并将它们作为参数传递给date函数来转换成date对象，用strftime()函数来将date对象转换成特定格式的字符串
                    date_cell = date(*date_cell[:3]).strftime('%Y/%m/%d')
                    map[worksheet.cell_value(0, col_index)] = date_cell
                else:
                    # 将sheet中非表示日期的值赋给non_date_celld对象
                    non_date_cell = worksheet.cell_value(row_index, col_index)
                    map[worksheet.cell_value(0, col_index)] = non_date_cell
            res.append(map)
    return col_name, res

if __name__ == "__main__":
    engine = create_engine("sqlite:///:memory:")
    metadata_obj = MetaData()

    # create city SQL table
    # table_name = "city_stats"
    # path = input("请输入数据所在路径：")
    # table_name = input("请输入表名：")
    # sheet = input("请输入sheet名：")

    path = 'test-data/'
    table_name = 'test'
    sheet = 'Sheet1'
    col_name, rows = read_excel(path, table_name, sheet)

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