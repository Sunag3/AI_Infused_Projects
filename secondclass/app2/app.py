import gradio as gr
from agent import agent

def chat(message, history):
    # Gradio fills the message and history return past resuts for you
    return agent(message)

gr.ChatInterface(
    fn=chat,
    title="Smart Shop Assistant",
    description="Ask me the price of shoes, hat, bag, shorts or pants - I will look it up",
).launch(share=True)