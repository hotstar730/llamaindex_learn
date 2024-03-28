import sys
from typing import List

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from llama_index.core.base.llms.types import ChatMessage
import os
import openai
# 把当前文件所在文件夹的父文件夹路径加入到PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_service.agents import *

from agents.agent_chat import AgentChat
from agents.agent_document import AgentDocument
from agents.agent_langchain import AgentLangChainSql

# 没有用到，不填会报错
from agents.agent_sql import AgentSql

os.environ["OPENAI_API_KEY"] = "sk-.."
openai.api_key = os.environ["OPENAI_API_KEY"]

# 不同的代理对象，用于聊天、查知识库、查数据表等
agent_chat = AgentChat()
agent_document = AgentDocument()
# agent_sql = AgentSql()
agent_sql = AgentLangChainSql()

user_chat_history = {}
user_chat_len = 10

# 启动 Flask server
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
def hello_world():
    return 'llm api server running!'


@app.route('/chat')
@cross_origin()
def chat():
    # message - 用户发送过来的消息
    # model   - 选择的模型，默认是qwen:7b-chat
    # user    - 用户id，用来标识用户
    # type    - 消息类型，1 聊天消息，2 数据查询，3 知识库
    # stream  - 是否流式返回，默认值为False
    query = request.args.get("query", type=str)
    userId = request.args.get('userId', type=str, default='default')
    type = request.args.get("type", type=int, default=1)
    model = request.args.get("model", type=str, default='qwen:7b-chat')
    stream = request.args.get("stream", type=bool, default=False)

    if query is None:
        return jsonify({"error": "query field is missing"}), 400

    # 通过user，获取用户聊天的消息记录
    if user_chat_history.get(userId) is None:
        user_chat_history[userId]: List[ChatMessage] = []
    chat_history = user_chat_history[userId]
    chat_history.append(ChatMessage(role="user", content=query))
    if len(chat_history) >= user_chat_len:
        chat_history = chat_history[len(chat_history) - user_chat_len:]

    # 通过type，判断调用的agent
    # 通过stream，调用不同的接口
    try:
        if type == 1:
            response = agent_chat.chat(chat_history)
            chat_history.append(response)
            return jsonify({"response": str(response.content)})
        elif type == 2:
            response = agent_document.chat(chat_history)
            chat_history.append(response)
            return jsonify({"response": str(response.content)})
        elif type == 3:
            response = agent_sql.chat(chat_history)
            chat_history.append(response)
            return jsonify({"response": str(response.content)})
        else:
            return jsonify({"error": "type field is incorrect"}), 400
    except:
        user_chat_history[userId] = []
        return jsonify({"error": "type field is incorrect"}), 400


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=11435)
