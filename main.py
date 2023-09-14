import os
from apikey import apikey
from langchain import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.chains.conversation.memory import ConversationBufferMemory

import streamlit as st

os.environ['OPENAI_API_KEY'] = apikey

def log_report_query(query):
    print("report:\n")
    print(query)

def log_faq_query(query):
    print("FAQ:\n")
    print(query)

def log_action_query(query):
    print("action:\n")
    print(query)

tools = [
    Tool(
        name = "Run Report",
        func = lambda query: log_report_query(query),
        description = "use when you want specific answers about a company, employees or other specialized query",
    ),
    Tool(
        name = "Answer FAQ",
        func = lambda query: log_faq_query(query),
        description = "use to answer a general question that can be backed up by support documentation",
    ),
    Tool(
        name = "Run Action",
        func = lambda query: log_action_query(query),
        description = "use when you need to carry out an action",
    ),
]

memory = ConversationBufferMemory(memory_key="chat_history")
if "agent_memory" not in st.session_state:
    st.session_state["agent_memory"] = memory

llm = OpenAI(temperature=0, verbose=True)
agent_chain = initialize_agent(tools, llm, agent="conversational-react-description", memory=st.session_state["agent_memory"], verbose=True)

st.header(":blue[Langchain chatbot with agent/tools and memory ]")
user_input = st.text_input("You: ")

if st.button("submit:"):

    st.markdown(agent_chain.run(input=user_input))

    st.session_state["agent_memory"] += memory.buffer
    print(st.session_state["agent_memory"])
