from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant. Use the conversation history to stay consistent"),
    MessagesPlaceholder('history'),
    ("human", "{question}")
])

chain = prompt | model


# for a demo we will hardcode the history here just to demonstrate
history = [
    HumanMessage("My Name is Arav"),
    AIMessage("Hi Arav!.")
]

answer = chain.invoke(
    {"history": history,
    "question": "Whats my sweet name?"}
)

print(answer.content)

