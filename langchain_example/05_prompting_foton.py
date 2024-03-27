from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.llms.ollama import Ollama
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.embeddings import HuggingFaceEmbeddings
import sys
sys.path.append('..')
from llm_service.util.mysql import MysqlUtil

db = SQLDatabase.from_uri("mysql+pymysql://root:Foton12345&@1.92.64.112/llama", sample_rows_in_table_info=3)

llm = Ollama(model="pxlksr/defog_sqlcoder-7b-2:Q8")
llm.temperature = 0
llm.base_url = "http://1.92.64.112:11434"
chain = create_sql_query_chain(llm, db)

# 从配置表中读取提示信息
mysql = MysqlUtil(host='1.92.64.112', db='llama_config', user='root', passwd='Foton12345&')
sql = "select * from prompts"
results = mysql.get(sql)
examples = []
for res in results:
    prompt = {}
    prompt['input'] = res['input']
    prompt['query'] = res['query']
    examples.append(prompt)
print(examples)

example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")
prompt = FewShotPromptTemplate(
    examples=examples[:5],
    example_prompt=example_prompt,
    prefix="You are a MySQL expert. Given an input question, create a syntactically correct SQLite query to run. Unless otherwise specificed, do not return more than {top_k} rows.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries.",
    suffix="User input: {input}\nSQL query: ",
    input_variables=["input", "top_k", "table_info"],
)

modelPath = '/Users/amalyok/Desktop/foton/llamaindex_learn/langchain_example/bge-small-en-v1.5'
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embeddings,
    FAISS,
    k=5,
    input_keys=["input"],
)

_input = {"question": "终端离线时长小于10的车辆有多少?"}
ret = chain.invoke(_input)
print(ret)
insert_sql = "insert into llm_query_result(input, query) values (%s, %s)"
params = [_input.get('question'), ret]
mysql.crud(insert_sql, params)
print(db.run(ret))