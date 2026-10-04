from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from scraper import fetch_website_contents
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


# resuable prompt with {website} as a placeholder for the website content
prompt = ChatPromptTemplate.from_template(
    "You analyze the contents of a website and give a short, friendly summary. "
    "Ignore navigation menus. Respond in markdown format, and keep it concise. "
    "Do not include any HTML tags or code snippets in your response. "
    "Give in table like data"
    "Website content: {website}"
)

# Same model we used in the previous example, but now using LangChain's OpenAI wrapper
model = ChatOpenAI(model_name="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7, max_tokens=300)

# parser package-opener: hands back plain text, no JSON or other structured output
parser = StrOutputParser()

# snap thm together into a chain, so we can call it with just the URL and get the summary back
chain = prompt | model | parser  # Chain the prompt, model, and parser together

def summarize(url):
    return chain.invoke({'website': fetch_website_contents(url)})  # Fetch website contents and invoke the chain


if __name__ == "__main__":
    test_url = "https://www.scaler.com/"  # Replace with a real URL for testing
    summary = summarize(test_url)
    print(summary)
