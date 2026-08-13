from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from dotenv import load_dotenv

# Load environment variables before importing deepeval/langchain
load_dotenv()

# Initializing language model
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=500, temperature=0.5)

#simple in memory store for chat histories
store = {}

def get_chat_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an ai engineering expert"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="history"
)

session_id = "user_123"

#Wrap the chain with message history
response1 = chain_with_history.invoke(
    {"input": "What does a basic AI agent do?"},
    config={"configurable": {"session_id": session_id}}
)
print("AI:", response1.content)

response2 = chain_with_history.invoke(
    {"input": "What was my previous message?"},
    config={"configurable": {"session_id": session_id}}
)
print("AI:", response2.content)

print("\nConversation History:")
for message in store[session_id].messages:
    print(f"{message.type}: {message.content}")