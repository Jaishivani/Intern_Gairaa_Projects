# Gairaa Intern Projects

This repository is a collection of AI, ML, and LLM-focused projects built as part of internship work and learning exercises. The projects cover chatbot systems, document intelligence, computer vision, workflow automation, and utility AI apps.

## Repository overview

| Project | Type | Summary |
| --- | --- | --- |
| [Bank_chatbot](./Bank_chatbot) | FastAPI + LLM | Banking chatbot with login, protected account queries, and Groq-powered responses. |
| [Study_buddy](./Study_buddy) | RAG study assistant | PDF-based study assistant with document Q&A, summaries, quizzes, citations, and web fallback. |
| [Path_hole_XAI](./Path_hole_XAI) | Computer vision | Road pothole classifier with Grad-CAM explainability and model evaluation. |
| [kitchen_ass_langgrap](./kitchen_ass_langgrap) | LangGraph assistant | Kitchen helper that decides when web info is needed and generates recipe guidance. |
| [text_sumz](./text_sumz) | FastAPI summarizer | Rule-based extractive text summarization API without external model dependencies. |
| [Medical_rep_assist](./Medical_rep_assist) | Medical assistant | Backend for a medical representative or knowledge assistant workflow. |
| [Weather_Agent](./Weather_Agent) | Weather assistant | Lightweight weather-related AI agent script. |
| [ML_vs_DL_Combined.ipynb](./ML_vs_DL_Combined.ipynb) | Notebook | Notebook comparing machine learning and deep learning concepts. |
| [Semantica_Project_Brief.pptx](./Semantica_Project_Brief.pptx) | Presentation | Project brief / presentation deck for one of the workstreams. |

## Highlights

- Conversational AI and LLM workflows using FastAPI, LangGraph, and Groq.
- Retrieval-augmented generation for document understanding and study assistance.
- Computer vision with transfer learning and explainability using Grad-CAM.
- Practical learning projects focused on AI application design, evaluation, and deployment basics.
- Projects structured with Python virtual environments, requirement files, and environment-variable configuration.

## Project details

### 1. Bank Chatbot

Location: [Bank_chatbot](./Bank_chatbot)

A banking chatbot demo built with FastAPI and Groq. It includes:
- token-based login
- protected banking endpoints
- input guardrails and sensitive-data blocking
- Excel-backed demo data
- Docker support

### 2. Study Buddy

Location: [Study_buddy](./Study_buddy)

A local study assistant for PDF-based learning. It supports:
- document upload and indexing
- Q&A with page citations
- summaries and quizzes
- topic exploration
- web fallback when the answer is not grounded in the document

### 3. Path Hole XAI

Location: [Path_hole_XAI](./Path_hole_XAI)

A pothole detection project that classifies road images and explains predictions with Grad-CAM. It emphasizes:
- image classification using ResNet18
- class-balanced training
- evaluation metrics and confusion matrix
- visual explanation of model decisions

### 4. Kitchen Assistant with LangGraph

Location: [kitchen_ass_langgrap](./kitchen_ass_langgrap)

A command-line helper for kitchen questions, with routing logic to decide whether live web information is needed before producing a final answer.

### 5. Text Summarizer API

Location: [text_sumz](./text_sumz)

A simple FastAPI summarization service that uses an extractive sentence-scoring approach instead of a large ML model.

### 6. Medical Representative Assistant

Location: [Medical_rep_assist](./Medical_rep_assist)

A backend-focused assistant project for medical data workflows and document-based assistance.

### 7. Weather Agent

Location: [Weather_Agent](./Weather_Agent)

A lightweight weather-related AI agent implementation built as a standalone Python script.

## Getting started

Each project is independent and may have its own environment variables and dependency setup. In most cases:

1. Open the project directory.
2. Create and activate a virtual environment.
3. Install dependencies from the project's `requirements.txt` file.
4. Copy `.env.example` to `.env` if available and set the required API keys.
5. Run the project using its own instructions.

Typical Windows commands:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For project-specific run steps, refer to each sub-project's README.

## Environment and security notes

- Environment files such as `.env` are excluded from Git via the root `.gitignore`.
- Most LLM-based projects require API keys from providers such as Groq or Tavily.
- Use project-level `.env.example` files as templates before creating your own local `.env` files.

## Suggested usage

This repository is best used as:
- a learning portfolio
- a collection of AI experiments
- a reference for FastAPI, LangGraph, RAG, and computer vision patterns
- a starting point for extending or combining features across projects

## Note

Some folders are early-stage experiments or support projects. For exact setup and execution details, read the README in the specific project folder you want to run.
