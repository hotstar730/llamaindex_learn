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

from langchain.globals import set_debug



# class AgentLangChainSql:
#
#     def __init__(self) -> None:
#         print("1")
#
#     def chat(self, messages: List[ChatMessage]) -> ChatMessage:
#         if messages:
#             message = messages[-1]
#             self.query_engine.update_prompts('use mysql database')
#             response = self.query_engine.query(message.content + "")
#             return ChatMessage(role="assistant", content=response)
#         else:
#             return ChatMessage(role="assistant", content="can i help you!")

from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.llms.ollama import Ollama
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

db = SQLDatabase.from_uri("mysql+pymysql://root:Foton12345&@1.92.64.112/llama", sample_rows_in_table_info=3)

# 1 Dialect-specific prompting
llm = Ollama(model="pxlksr/defog_sqlcoder-7b-2:Q8")
llm.temperature = 0
llm.base_url = "http://1.92.64.112:11434"
chain = create_sql_query_chain(llm, db)

# 2 Table definitions and example rows
context = db.get_context()

# 3 Few-shot examples
examples = [
    {"input": "List all artists.", "query": "SELECT * FROM Artist;"},
    {
        "input": "Find all albums for the artist 'AC/DC'.",
        "query": "SELECT * FROM Album WHERE ArtistId = (SELECT ArtistId FROM Artist WHERE Name = 'AC/DC');",
    },
]

# 4 Dynamic few-shot examples
modelPath = 'bge-small-en-v1.5'
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,        # Provide the pre-trained model's path
    model_kwargs=model_kwargs,   # Pass the model configuration options
    encode_kwargs=encode_kwargs  # Pass the encoding options
)

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embeddings,
    FAISS,
    k=5,
    input_keys=["input"],
)

example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")
prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="You are a MySQL expert. Given an input question, create a syntactically correct SQLite query to run. Unless otherwise specificed, do not return more than {top_k} rows.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries.",
    suffix="User input: {input}\nSQL query: ",
    input_variables=["input", "top_k", "table_info"],
)


chain = create_sql_query_chain(llm, db, prompt)
# chain.get_prompts()[0].partial(table_info=context["table_info"])
set_debug(True)
ret = chain.invoke({"question": "how many artists are there?"})
print(ret)
