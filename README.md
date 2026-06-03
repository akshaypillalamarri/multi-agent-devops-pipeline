---
title: Multi-Agent DevOps Pipeline
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.29.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Multi-Agent DevOps Pipeline

A production-grade **multi-agent AI system** that takes any coding task through a complete DevOps pipeline using **LangGraph** and **LangChain**.

**Live Demo:** [Try it on Hugging Face Spaces](https://huggingface.co/spaces/akshayrinku/multi-agent-devops-pipeline)

## 4 Specialized AI Agents

```
Your Task
    ↓
[✍️ Code Writer Agent] — writes production-grade code
    ↓
[🔍 Code Reviewer Agent] — reviews bugs, security, performance
    ↓
[🧪 Test Writer Agent] — writes comprehensive unit tests
    ↓
[🚀 DevOps Agent] — creates Dockerfile, CI/CD pipeline, deployment guide
    ↓
Complete Production-Ready Package
```

## Features

- **Code Writer** — writes clean, production-grade code with best practices
- **Code Reviewer** — reviews for bugs, security vulnerabilities, performance issues
- **Test Writer** — generates comprehensive unit tests with pytest/JUnit/Jest
- **DevOps Agent** — creates Dockerfile, GitHub Actions CI/CD, deployment guide
- **5 Sample Tasks** — pre-loaded examples for Python, JavaScript, Java
- **Tabbed Results** — each agent's output in its own tab

## Tech Stack

| Layer | Technology |
| --- | --- |
| Agent Orchestration | LangGraph |
| LLM Framework | LangChain |
| LLM | Google Gemini 2.5 Flash |
| UI | Gradio |
| Deployment | Hugging Face Spaces |
| Language | Python |

## How to Run Locally

```bash
git clone https://github.com/akshaypillalamarri/multi-agent-devops-pipeline
cd multi-agent-devops-pipeline
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
python app.py
```

## About Me

Akshay Pillalamarri — AI/ML Engineer based in Folsom, CA.
LangChain + LangGraph Certified. AWS Certified.

- Portfolio: akshaypillalamarri.github.io
- LinkedIn: linkedin.com/in/akshay-pillalamarri
- GitHub: github.com/akshaypillalamarri

## License

MIT
