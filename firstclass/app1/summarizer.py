from openai import OpenAI
from dotenv import load_dotenv
import os

from scraper import fetch_website_contents

load_dotenv() # thi reads the .env file and loads the environment variables

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = "You analyze the contents of website and give a short, friendly summary. Ignore navigation menus, Respond in markdown format, and keep it concise. Do not include any HTML tags or code snippets in your response."


def summarize(url):
    website = fetch_website_contents(url)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Summarize the following website content:\n\n{website}"}
        ],
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()