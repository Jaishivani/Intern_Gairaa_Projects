# Kitchen Assistant

A command-line kitchen assistant built with Python and LangGraph. It uses the NVIDIA API for language generation and Tavily for current web information.

## Requirements

- Python 3.10 or newer
- NVIDIA API key
- Tavily API key

## Setup

Open PowerShell in the project directory:

```powershell
cd C:\Users\welcome\Downloads\kitchen_ass_langgrap
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create `.env` from `.env.example` and set your credentials:

```env
NVIDIA_API_KEY=your_nvidia_api_key
TAVILY_API_KEY=your_tavily_api_key
NVIDIA_MODEL=nvidia/llama-3.1-nemotron-70b-instruct
```

The `.env` file is ignored by Git and must not be committed.

## Run

From the project directory, with the virtual environment active:

```powershell
python main.py
```

Then enter a kitchen question, for example:

```text
You: I have rice, eggs, and tomatoes. What can I cook?
```

Press `Ctrl+C` to exit.

## Graph Flow

```text
START
  -> input
  -> llm
  -> conditional routing
       -> output -> END
       -> web_search -> llm -> output -> END
```

The first LLM pass decides whether current web information is needed. Tavily is called only for web-required questions. A second NVIDIA pass uses the Tavily results to produce the final answer, and the application appends verified Tavily source links to web-backed responses.

## Project Structure

```text
app/
  config.py
  state.py
  graph.py
  nodes/
    input_node.py
    llm_node.py
    web_search_node.py
    output_node.py
  services/
    nvidia.py
    tavily.py
main.py
requirements.txt
.env.example
.gitignore
```

This project uses LangGraph directly and does not use LangChain.
