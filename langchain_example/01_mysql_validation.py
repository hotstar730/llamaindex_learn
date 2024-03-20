#!/usr/bin/python3
# encoding=utf-8

"""
    项目名称:  llamaindex_learn
    文件名称： 01_mysql_exmple.py
    功能描述： ..
    创建者：   lixinxin
    创建日期： 2024/3/20 9:14
"""
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.llms.ollama import Ollama
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

db = SQLDatabase.from_uri("mysql+pymysql://root:Foton12345&@1.92.64.112/llama")

# 定义你的LLM
llm = Ollama(model="pxlksr/defog_sqlcoder-7b-2:Q8")
llm.temperature = 0
llm.base_url = "http://1.92.64.112:11434"

chain = create_sql_query_chain(llm, db)

system = """Double check the user's {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

Output the final SQL query only."""
prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "{query}")]
).partial(dialect=db.dialect)
validation_chain = prompt | llm | StrOutputParser()

# result = chain.invoke({"question": "在test表中车辆总数有多少?"})
# print(result)

full_chain = {"query": chain} | validation_chain
# full_chain = {"query": chain}

query = full_chain.invoke(
    {
        "question": "在test表中车辆总数有多少?"
    }
)
query
ret = db.run(query)
ret

