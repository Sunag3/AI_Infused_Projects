import os
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
groq_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # Replace with your Groq API key if different
)


def ask(client, model, prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


def battle(prompt):
    a = ask(openai_client, "gpt-4o-mini", prompt)
    b = ask(groq_client, "gpt-5-nano", prompt)
    return a, b


def vote(action):
    return f"Feedback recorded: {action}"


with gr.Blocks() as demo:
    gr.Markdown("## AI Model Battle Arena")
    gr.Markdown("LLM Arena - one prompt, two models")
    
    prompt = gr.Textbox(label="Enter your prompt", lines=4)
    go = gr.Button("Battle !", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Model A: gpt-4o-mini")
            out_a = gr.Markdown()
            with gr.Row():
                up_a = gr.Button("👍 Vote for A")
                down_a = gr.Button("👎 Vote against A")
        
        with gr.Column():
            gr.Markdown("### Model B: llama-3.3-70b-versatile")
            out_b = gr.Markdown()
            with gr.Row():
                up_b = gr.Button("👍 Vote for B")
                down_b = gr.Button("👎 Vote against B")

    verdict = gr.Markdown()

    # Event Handlers
    go.click(battle, inputs=prompt, outputs=[out_a, out_b])
    
    up_a.click(lambda: vote("Upvoted Model A"), outputs=verdict)
    down_a.click(lambda: vote("Downvoted Model A"), outputs=verdict)
    up_b.click(lambda: vote("Upvoted Model B"), outputs=verdict)
    down_b.click(lambda: vote("Downvoted Model B"), outputs=verdict)

demo.launch(share=True)