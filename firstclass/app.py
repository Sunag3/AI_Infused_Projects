import gradio as gr
from summarizer import summarize

gr.Interface(fn=summarize,
    inputs=gr.Textbox(lines=10, placeholder="Enter the URL of the website to summarize..."),
    outputs=gr.Textbox(lines=10),
    title="Website Summarizer",
    description="Enter the URL of a website, and this tool will fetch its contents and provide a concise summary using OpenAI's GPT-4o-mini model."
).launch(share=True) # Share=True -> public link you can share with others