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
from langchain.agents import AgentType
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.llms.ollama import Ollama
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

# 现在，db位于我们的目录中，我们可以使用 SQLAlchemy 驱动的SQLDatabase类与其进行交互：
db = SQLDatabase.from_uri("mysql+pymysql://root:Foton12345&@1.92.64.112/test")
context = db.get_context()
print(list(context))
print(context["table_info"])
# print(db.dialect)
# print(db.get_usable_table_names())
# print(db.get_table_info())
# result = db.run("SELECT * FROM Artist LIMIT 10;")
# print(result)

# 我们将使用 聊天模型和"openai-tools"代理，代理将使用 OpenAI 的函数调用 API 来驱动代理的工具选择和调用。
llm = Ollama(model="pxlksr/defog_sqlcoder-7b-2:Q8")
llm.temperature = 0
llm.base_url = "http://1.92.64.112:11434"

chain = create_sql_query_chain(llm, db)


print("1=====================================================")
prompt_with_context = chain.get_prompts()[0].partial(table_info=context["table_info"])
print(prompt_with_context.pretty_repr())

response = chain.invoke({"question": "List the total sales per country. Which country's customers spent the most?"})
print("1=====================================================")
print(response)
print("2=====================================================")
prompts = chain.get_prompts()
print(chain.get_prompts())

from langchain_community.agent_toolkits import create_sql_agent
# agent_executor = create_sql_agent(llm, db=db, agent_type="zero-shot-react-description", return_intermediate_steps=True, verbose=True, agent_executor_kwargs={"handle_parsing_errors": True})
agent_executor = create_sql_agent(llm, db=db, verbose=True, agent_executor_kwargs={"handle_parsing_errors": True})

response = agent_executor.invoke(
    "how many artists are there?"
)
print(response)
exit(0)
