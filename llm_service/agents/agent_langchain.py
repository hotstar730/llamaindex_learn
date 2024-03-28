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
from typing import List, Union, Dict, Any
from langchain.chains.sql_database.query import create_sql_query_chain, SQLInput, SQLInputWithTables
from langchain.globals import set_debug
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_experimental.sql import SQLDatabaseChain
from llama_index.core.base.llms.types import ChatMessage
from operator import itemgetter

from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, Runnable

from llm_service.util.mysql_util import MysqlUtil


class AgentLangChainSql:
    _example_selector: SemanticSimilarityExampleSelector
    _mysql: MysqlUtil
    _chain_query: Runnable[Union[SQLInput, SQLInputWithTables, Dict[str, Any]], str]
    _db_execute: QuerySQLDataBaseTool
    _chain_answer: Runnable[Union[SQLInput, SQLInputWithTables, Dict[str, Any]], str]
    _debug: False

    def __init__(self, debug=False) -> None:
        self._debug = debug
        if self._debug:
            set_debug(True)

        self._mysql = MysqlUtil(host='1.92.64.112', db='llama_config', user='root', passwd='Foton12345&')

        # 1 Dialect-specific prompting
        llm = Ollama(model="pxlksr/defog_sqlcoder-7b-2:Q8")
        llm.temperature = 0
        llm.base_url = "http://1.92.64.112:11434"

        # 2 Table definitions and example rows
        db = SQLDatabase.from_uri("mysql+pymysql://root:Foton12345&@1.92.64.112/llama", sample_rows_in_table_info=3)

        # 3 get prompt
        prompt = self._get_prompt()

        self._chain_query = create_sql_query_chain(llm, db, prompt)
        self._db_execute = QuerySQLDataBaseTool(db=db)

        answer_prompt = PromptTemplate.from_template(
            """Given the following user question, corresponding SQL query, and SQL result, answer the user question.
                Question: {question}
                SQL Query: {query}
                SQL Result: {result}
                Answer: """
        )
        self._chain_answer = answer_prompt | llm | StrOutputParser()


    def _db_get_examples(self) -> []:
        # 从配置表中读取提示信息
        sql = "select * from prompts"
        results = self._mysql.get(sql)
        examples = []
        for res in results:
            examples.append({'input': res['input'], 'query': res['query']})
        return examples

    def _db_save_query_result(self, input_msg: str, query_msg: str, result_msg: str, state: int) -> any:
        insert_sql = "insert into llm_query_result(input, query, result, state) values (%s, %s, %s, %s)"
        return self._mysql.crud(insert_sql, [input_msg, query_msg, result_msg, str(state)])

    def _get_prompt(self) -> FewShotPromptTemplate:
        # 3.1 Few-shot examples
        examples = self._db_get_examples()

        # 3.2 Dynamic few-shot examples
        modelPath = 'data/embed_model/bge-small-en-v1.5'
        if self._debug:
            modelPath = '../data/embed_model/bge-small-en-v1.5'
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        embeddings = HuggingFaceEmbeddings(
            model_name=modelPath,  # Provide the pre-trained model's path
            model_kwargs=model_kwargs,  # Pass the model configuration options
            encode_kwargs=encode_kwargs  # Pass the encoding options
        )
        self._example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples,
            embeddings,
            FAISS,
            k=5,
            input_keys=["input"],
        )

        # 3.3 prompt
        example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")
        prompt = FewShotPromptTemplate(
            example_selector=self._example_selector,
            example_prompt=example_prompt,
            prefix="You are a MySQL expert. Given an input question, create a syntactically correct Mysql query to run. Unless otherwise specificed, do not return more than {top_k} rows. then look at the results of the query and return the answer.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries.",
            suffix="User input: {input}\nSQL query: ",
            input_variables=["input", "top_k", "table_info"],
        )
        return prompt

    def _get_tips_example(self, message) -> str:
        response_example = ""
        examples = self._example_selector.select_examples({"input": message})
        for i, j in enumerate(examples):
            response_example += '\n' + str(i + 1) + "." + j['input']
        print(examples)
        return response_example

    def chat(self, messages: List[ChatMessage]) -> ChatMessage:
        if not messages:
            return ChatMessage(role="assistant", content="can i help you!")
        message = messages[-1].content

        try:
            question = message
            query = self._chain_query.invoke({"question": question})
            result = self._db_execute.run(query)
            response = self._chain_answer.invoke({"question": question, "query": query, "result": result})
            self._db_save_query_result(message, query, response, 1)
        except:
            response = '暂无答案，请换个问题试试。eg：' + self._get_tips_example(message)
            self._db_save_query_result(message, "", response, 2)
        return ChatMessage(role="assistant", content=response)


agent_lang_chain = AgentLangChainSql(True)
# ret = agent_lang_chain.chat([ChatMessage(role="assistant", content='盘点车辆总数')])
ret = agent_lang_chain.chat([ChatMessage(role="assistant", content='福田总部在哪')])
print(ret)
