# 🤖 Agentic AI Website Summarizer

An intelligent **Agentic AI** application that takes a website URL from a user, autonomously fetches and scrapes its content, and generates a structured summary. Built using **Python**, **Gradio** for the user interface, and powered by **OpenAI models**.

---

## ✨ Features

* **URL Processing:** Simply paste any website link.
* **Agentic Execution:** The AI agent analyzes the page, identifies key sections, and extracts critical information.
* **Smart Summarization:** Delivers a clear, high-quality summary rather than just a wall of text.
* **Interactive UI:** A clean and modern user interface powered by Gradio.

---

## 🛠️ Tech Stack

* **Language:** Python 3.8+
* **Framework:** Gradio (for the web interface)
* **LLM Orchestration:** OpenAI API
* **Web Scraping:** Beautiful Soup / Requests / Trafilatura

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com
cd agentic-ai-summarizer
```

### 2. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```
*(If you don't have a `requirements.txt` yet, install the core packages manually: `pip install gradio openai beautifulsoup4 requests`)*

### 3. Set Up Your OpenAI API Key
Create a `.env` file in the root directory or export your key directly in your terminal:

**On Linux/macOS:**
```bash
export OPENAI_API_KEY="your-actual-api-key-here"
```

**On Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY="your-actual-api-key-here"
```

### 4. Run the Application
```bash
python app.py
```
Once running, click the local URL provided in your terminal (usually `http://127.0.0.1:7860`) to open the app in your browser!

---

## 💡 How It Works

1. **User Input:** The user provides a web link through the Gradio text box.
2. **Agent Fetching:** The Python backend scrapes the text content, filtering out unnecessary HTML clutter like ads and navigation links.
3. **LLM Analysis:** The text is passed to the OpenAI model with custom instructions to act as an expert research agent.
4. **Output:** The agent returns a beautifully formatted markdown summary directly to the Gradio interface.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
