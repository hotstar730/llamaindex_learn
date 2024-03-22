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

db = SQLDatabase.from_uri("mysql+pymysql://root:Foton12345&@1.92.64.112/llama", sample_rows_in_table_info=3)
print(db.dialect)
print(db.get_usable_table_names())
ret = db.run("SELECT * FROM test LIMIT 10;")

# 1 Dialect-specific prompting
llm = Ollama(model="pxlksr/defog_sqlcoder-7b-2:Q8")
llm.temperature = 0
llm.base_url = "http://1.92.64.112:11434"
chain = create_sql_query_chain(llm, db)
chain.get_prompts()[0].pretty_print()

# 2 Table definitions and example rows
context = db.get_context()
print("1=====================================================")
print(list(context))
print(context["table_info"])

# insert the entirety of this information in our prompt
prompt_with_context = chain.get_prompts()[0].partial(table_info=context["table_info"])
print("2=====================================================")
print(prompt_with_context.pretty_repr()[:1500])
print("3=====================================================")
print(chain.get_prompts()[0].pretty_print())

# 3 Few-shot examples
examples = [
    {"input": "List all artists.", "query": "SELECT * FROM Artist;"},
    {
        "input": "Find all albums for the artist 'AC/DC'.",
        "query": "SELECT * FROM Album WHERE ArtistId = (SELECT ArtistId FROM Artist WHERE Name = 'AC/DC');",
    },
    {
        "input": "List all tracks in the 'Rock' genre.",
        "query": "SELECT * FROM Track WHERE GenreId = (SELECT GenreId FROM Genre WHERE Name = 'Rock');",
    },
    {
        "input": "Find the total duration of all tracks.",
        "query": "SELECT SUM(Milliseconds) FROM Track;",
    },
    {
        "input": "List all customers from Canada.",
        "query": "SELECT * FROM Customer WHERE Country = 'Canada';",
    },
    {
        "input": "How many tracks are there in the album with ID 5?",
        "query": "SELECT COUNT(*) FROM Track WHERE AlbumId = 5;",
    },
    {
        "input": "Find the total number of invoices.",
        "query": "SELECT COUNT(*) FROM Invoice;",
    },
    {
        "input": "List all tracks that are longer than 5 minutes.",
        "query": "SELECT * FROM Track WHERE Milliseconds > 300000;",
    },
    {
        "input": "Who are the top 5 customers by total purchase?",
        "query": "SELECT CustomerId, SUM(Total) AS TotalPurchase FROM Invoice GROUP BY CustomerId ORDER BY TotalPurchase DESC LIMIT 5;",
    },
    {
        "input": "Which albums are from the year 2000?",
        "query": "SELECT * FROM Album WHERE strftime('%Y', ReleaseDate) = '2000';",
    },
    {
        "input": "How many employees are there",
        "query": 'SELECT COUNT(*) FROM "Employee"',
    },
]

from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")
prompt = FewShotPromptTemplate(
    examples=examples[:5],
    example_prompt=example_prompt,
    prefix="You are a MySQL expert. Given an input question, create a syntactically correct SQLite query to run. Unless otherwise specificed, do not return more than {top_k} rows.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries.",
    suffix="User input: {input}\nSQL query: ",
    input_variables=["input", "top_k", "table_info"],
)
print("4=====================================================")
print(prompt.format(input="How many artists are there?", top_k=3, table_info="foo"))

# 4 Dynamic few-shot examples
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.embeddings import HuggingFaceEmbeddings

modelPath = 'bge-small-en-v1.5'
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

ret = example_selector.select_examples({"input": "how many artists are there?"})
print("5=====================================================")
print(ret)

prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="You are a MySQL expert. Given an input question, create a syntactically correct SQLite query to run. Unless otherwise specificed, do not return more than {top_k} rows.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries.",
    suffix="User input: {input}\nSQL query: ",
    input_variables=["input", "top_k", "table_info"],
)
print(prompt.format(input="how many artists are there?", top_k=3, table_info="foo"))

chain = create_sql_query_chain(llm, db, prompt)
chain.get_prompts()[0].partial(table_info=context["table_info"])
ret = chain.invoke({"question": "how many artists are there?"})
print(ret)
