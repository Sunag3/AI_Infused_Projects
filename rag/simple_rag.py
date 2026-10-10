from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# create a embedder
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db= Chroma(persist_directory='./chroma_db', embedding_function=embedder)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template(
    """Answer the question using only the context below.
    If the content doesnt contain the answer. say "I dont know".
    Be consise and quote facts directly

    context:
    {context}

    Question: {question}
    
    """
)


def rag_answer(question):
    chunks = db.similarity_search(question, k=3)
    context = "\n\n".join(c.page_content for c in chunks)
    chain = prompt | model
    return chain.invoke({"context": context, "question": question}).content


print(rag_answer("how many days for the credit to officially reflect on your bank or credit card?"))