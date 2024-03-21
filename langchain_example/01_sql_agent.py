#!/usr/bin/python3
# encoding=utf-8

"""
    项目名称:  llamaindex_learn
    文件名称： 01_mysql_exmple.py
    功能描述： ..
    创建者：   lixinxin
    创建日期： 2024/3/20 9:14
    验证成功。
"""
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.llms.ollama import Ollama
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

db = SQLDatabase.from_uri("mysql+pymysql://root:Foton12345&@1.92.64.112/llama")
# print(db.dialect)
# print(db.get_usable_table_names())
# result = db.run("SELECT * FROM test LIMIT 10;")
# print(result)

# 定义你的LLM
llm = Ollama(model="pxlksr/defog_sqlcoder-7b-2:Q8")
llm.temperature = 0
llm.base_url = "http://1.92.64.112:11434"

# db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
# db_chain.run("车辆总数有多少?")

from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)
# chain = write_query | execute_query
# print(chain.invoke({"question": "车辆总数有多少?"}))

# 创建一个简单的链，它接受一个问题，将其转换为 SQL 查询，执行查询，并使用结果来回答原始问题。
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: "Final answer here"
    """
    )

answer = answer_prompt | llm | StrOutputParser()
chain = (
        RunnablePassthrough.assign(query=write_query).assign(
            result=itemgetter("query") | execute_query
        )
        | answer
)
chain.get_prompts()[0].pretty_print()
result = chain.invoke({"question": "在test表中车辆总数有多少?"})
print(result)

print(chain.invoke({"question": "Describe the schema of the Invoice table"}))


