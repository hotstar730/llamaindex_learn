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
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain.globals import set_debug
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_experimental.sql import SQLDatabaseChain
from llama_index.core.base.llms.types import ChatMessage

from llm_service.util.mysql_util import MysqlUtil


class AgentLangChainSql:
    _prompt: FewShotPromptTemplate
    _llm: Ollama
    _db: SQLDatabase
    _example_selector: SemanticSimilarityExampleSelector
    _mysql: MysqlUtil

    def __init__(self) -> None:
        self._mysql = MysqlUtil(host='1.92.64.112', db='llama_config', user='root', passwd='Foton12345&')

        # 1 Dialect-specific prompting
        self._llm = Ollama(model="pxlksr/defog_sqlcoder-7b-2:Q8")
        self._llm.temperature = 0
        self._llm.base_url = "http://1.92.64.112:11434"

        # 2 Table definitions and example rows
        self._db = SQLDatabase.from_uri("mysql+pymysql://root:Foton12345&@1.92.64.112/llama", sample_rows_in_table_info=3)

        # 3 get prompt
        self._prompt = self._get_prompt()

        # get chain
        # self.chain = create_sql_query_chain(self._llm, self._db, self._prompt)
        self.chain = SQLDatabaseChain.from_llm(self._llm, self._db, self._prompt, verbose=False)

    def _db_get_examples(self) -> []:
        # 从配置表中读取提示信息
        sql = "select * from prompts"
        results = self._mysql.get(sql)
        examples = []
        for res in results:
            examples.append({'input': res['input'], 'query': res['query']})
        return examples

    def _db_save_query_result(self, input_msg: str, result_msg: str, state: int) -> any:
        insert_sql = "insert into llm_query_result(input, result, state) values (%s, %s, %s)"
        return self._mysql.crud(insert_sql, [input_msg, result_msg, str(state)])

    def _get_prompt(self) -> FewShotPromptTemplate:
        # 3.1 Few-shot examples
        examples = self._db_get_examples()

        # 3.2 Dynamic few-shot examples
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
            prefix="You are a MySQL expert. Given an input question, create a syntactically correct SQLite query to run. Unless otherwise specificed, do not return more than {top_k} rows.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries.",
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

        self.chain.return_sql = False
        message = messages[-1]
        try:
            response = self.chain.run(message)
            self._db_save_query_result(message, response, 1)
        except:
            response = '暂无答案，请换个问题试试。eg：' + self._get_tips_example(message)
            self._db_save_query_result(message, response, 2)
        return ChatMessage(role="assistant", content=response)

    def chat_debug(self, message: str) -> str:
        # 打开调试信息
        set_debug(True)
        # 返回sql
        self.chain.return_sql = False
        try:
            response = self.chain.run(message)
            self._db_save_query_result(message, response, 1)
        except:
            response = '暂无答案，请换个问题试试。eg：' + self._get_tips_example(message)
            self._db_save_query_result(message, response, 2)
        return response


agent_lang_chain = AgentLangChainSql()
ret = agent_lang_chain.chat_debug("盘点车辆总数")
# ret = agent_lang_chain.chat_debug("福田总部在哪")
print(ret)
