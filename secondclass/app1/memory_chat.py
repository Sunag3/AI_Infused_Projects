from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Load the Model First

model = ChatOpenAI(model="gpt-4o-mini")

# prompt
prompt = ChatPromptTemplate.from_messages([
    ('system', "You are a friendly assisant. Use the conversation history to stay consistent."),
    MessagesPlaceholder('history'),
    ("human", "{question}")
])

chain = prompt | model


history = []


print("Chat with the Bot Type ('quit' or 'exit').")
while True:
    question = input("You: ")
    if question.lower() in {'exit', 'quit'}:
        break

    answer = chain.invoke({
        "history": history,
        "question": question
    }).content
    print(f"BOT: {answer}")

    history.append(HumanMessage(question))
    history.append(AIMessage(answer))
    print(f" History now has {len(history)} messages")
    