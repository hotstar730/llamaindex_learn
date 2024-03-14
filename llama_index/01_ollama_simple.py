#!/usr/bin/python3
# encoding=utf-8

"""
    项目名称:  llamaindex_learn
    文件名称： 01_ollama_simple.py.py
    功能描述： ..
    创建者：   lixinxin
    创建日期： 2024/3/14 10:19
"""

from llama_index.llms.ollama import Ollama

if __name__ == "__main__":
    llm = Ollama(model="gemma:2b")
    llm.base_url = "http://1.92.64.112:11434"
    response = llm.complete("写一篇作文，题目是：我的妈妈?")
    print(response)
