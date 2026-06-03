"""
Multi-Agent DevOps Pipeline
Built with LangGraph + LangChain + Google Gemini
4 Specialized AI Agents: Code Writer → Code Reviewer → Test Writer → DevOps Agent
Author: Akshay Pillalamarri
GitHub: github.com/akshaypillalamarri
Portfolio: akshaypillalamarri.github.io
"""

import os
import json
import gradio as gr
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.3
)

# ─────────────────────────────────────────
# STATE
# ─────────────────────────────────────────
class DevOpsState(TypedDict):
    task: str
    language: str
    written_code: str
    review_feedback: str
    revised_code: str
    unit_tests: str
    deployment_config: str
    agent_log: list
    pipeline_complete: bool

# ─────────────────────────────────────────
# AGENT 1: CODE WRITER
# ─────────────────────────────────────────
def code_writer_agent(state: DevOpsState) -> DevOpsState:
    log = state.get("agent_log", [])
    log.append("✍️ Code Writer Agent: Writing code...")

    prompt = f"""You are a senior software engineer. Write clean, production-grade code.

Task: {state['task']}
Language: {state['language']}

Requirements:
- Write complete, working code
- Follow best practices and design patterns
- Add clear comments explaining key logic
- Handle edge cases and errors
- Make it production ready

Return ONLY the code with comments. No explanation outside the code."""

    response = llm.invoke([HumanMessage(content=prompt)])
    written_code = response.content.strip()
    log.append("✅ Code Writer: Code written successfully")

    return {**state, "written_code": written_code, "agent_log": log}


# ─────────────────────────────────────────
# AGENT 2: CODE REVIEWER
# ─────────────────────────────────────────
def code_reviewer_agent(state: DevOpsState) -> DevOpsState:
    log = state.get("agent_log", [])
    log.append("🔍 Code Reviewer Agent: Reviewing code...")

    prompt = f"""You are a senior code reviewer at a top tech company.

Review this {state['language']} code thoroughly:

{state['written_code']}

Provide a detailed review covering:
1. 🐛 BUGS — Any logic errors or potential runtime issues
2. 🔒 SECURITY — Vulnerabilities, injection risks, unsafe operations
3. ⚡ PERFORMANCE — Inefficiencies, unnecessary operations, optimization opportunities
4. 📖 READABILITY — Naming conventions, code clarity, documentation
5. 🏗️ ARCHITECTURE — Design patterns, SOLID principles, maintainability
6. ✅ OVERALL SCORE — Rate the code 1-10

Then provide the REVISED version of the code with all issues fixed.

Format your response as:
## CODE REVIEW
[Your detailed review]

## REVISED CODE
[The improved code]"""

    response = llm.invoke([HumanMessage(content=prompt)])
    full_response = response.content.strip()

    # Split review and revised code
    if "## REVISED CODE" in full_response:
        parts = full_response.split("## REVISED CODE")
        review_feedback = parts[0].replace("## CODE REVIEW", "").strip()
        revised_code = parts[1].strip()
    else:
        review_feedback = full_response
        revised_code = state['written_code']

    log.append("✅ Code Reviewer: Review complete")
    return {**state, "review_feedback": review_feedback, "revised_code": revised_code, "agent_log": log}


# ─────────────────────────────────────────
# AGENT 3: TEST WRITER
# ─────────────────────────────────────────
def test_writer_agent(state: DevOpsState) -> DevOpsState:
    log = state.get("agent_log", [])
    log.append("🧪 Test Writer Agent: Writing unit tests...")

    prompt = f"""You are a senior QA engineer specializing in test automation.

Write comprehensive unit tests for this {state['language']} code:

{state['revised_code']}

Requirements:
- Write tests using the appropriate testing framework:
  * Python: pytest
  * JavaScript: Jest
  * Java: JUnit 5
  * Other: appropriate framework
- Cover these test types:
  1. Happy path tests (normal expected behavior)
  2. Edge case tests (boundary conditions)
  3. Error handling tests (invalid inputs, exceptions)
  4. Integration tests where applicable
- Use descriptive test names that explain what's being tested
- Add comments explaining complex test scenarios
- Aim for 80%+ code coverage

Return ONLY the test code with comments."""

    response = llm.invoke([HumanMessage(content=prompt)])
    unit_tests = response.content.strip()
    log.append("✅ Test Writer: Unit tests written successfully")

    return {**state, "unit_tests": unit_tests, "agent_log": log}


# ─────────────────────────────────────────
# AGENT 4: DEVOPS AGENT
# ─────────────────────────────────────────
def devops_agent(state: DevOpsState) -> DevOpsState:
    log = state.get("agent_log", [])
    log.append("🚀 DevOps Agent: Creating deployment configuration...")

    prompt = f"""You are a senior DevOps engineer and cloud architect.

Create a complete deployment configuration for this {state['language']} application:

{state['revised_code']}

Provide ALL of the following:

1. DOCKERFILE — Production-ready containerization
2. GITHUB ACTIONS CI/CD PIPELINE — .github/workflows/deploy.yml
   - Trigger on push to main
   - Run tests
   - Build Docker image
   - Deploy to cloud
3. REQUIREMENTS/DEPENDENCIES FILE — requirements.txt, package.json, or pom.xml
4. ENVIRONMENT VARIABLES — .env.example with all required variables
5. DEPLOYMENT CHECKLIST — Step by step deployment guide

Format each section clearly with headers.
Make everything production-ready and follow DevOps best practices."""

    response = llm.invoke([HumanMessage(content=prompt)])
    deployment_config = response.content.strip()
    log.append("✅ DevOps Agent: Deployment configuration created")
    log.append("🎉 Pipeline Complete! All 4 agents finished successfully.")

    return {**state, "deployment_config": deployment_config, "pipeline_complete": True, "agent_log": log}


# ─────────────────────────────────────────
# BUILD LANGGRAPH PIPELINE
# ─────────────────────────────────────────
def build_pipeline():
    graph = StateGraph(DevOpsState)

    graph.add_node("code_writer", code_writer_agent)
    graph.add_node("code_reviewer", code_reviewer_agent)
    graph.add_node("test_writer", test_writer_agent)
    graph.add_node("devops", devops_agent)

    graph.set_entry_point("code_writer")
    graph.add_edge("code_writer", "code_reviewer")
    graph.add_edge("code_reviewer", "test_writer")
    graph.add_edge("test_writer", "devops")
    graph.add_edge("devops", END)

    return graph.compile()

pipeline = build_pipeline()


# ─────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────
def run_pipeline(task: str, language: str):
    if not task.strip():
        return "Please describe your coding task.", "", "", "", "", ""
    if not GEMINI_API_KEY:
        return "API key not configured.", "", "", "", "", ""

    initial_state = DevOpsState(
        task=task,
        language=language,
        written_code="",
        review_feedback="",
        revised_code="",
        unit_tests="",
        deployment_config="",
        agent_log=[],
        pipeline_complete=False
    )

    result = pipeline.invoke(initial_state)

    agent_log = "\n".join(result.get("agent_log", []))

    return (
        result.get("written_code", ""),
        result.get("review_feedback", ""),
        result.get("revised_code", ""),
        result.get("unit_tests", ""),
        result.get("deployment_config", ""),
        agent_log
    )


# ─────────────────────────────────────────
# SAMPLE TASKS
# ─────────────────────────────────────────
SAMPLES = {
    "🐍 Python REST API": "Build a REST API endpoint that accepts a user's name and email, validates the input, stores it in a dictionary, and returns a JSON response with a unique user ID",
    "☕ Java Calculator": "Build a calculator class with add, subtract, multiply, divide operations with proper exception handling for division by zero",
    "🟨 JavaScript Auth": "Build a user authentication function that validates email format, checks password strength (min 8 chars, uppercase, number, special char), and returns a JWT token",
    "🐍 Python Data Pipeline": "Build a data processing pipeline that reads a CSV file, filters rows based on a condition, calculates statistics, and outputs a summary report",
    "☕ Java Linked List": "Implement a singly linked list in Java with add, remove, search, and reverse operations"
}


# ─────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────
with gr.Blocks(title="Multi-Agent DevOps Pipeline", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🤖 Multi-Agent DevOps Pipeline
    ### Powered by LangGraph + LangChain + Google Gemini | Built by Akshay Pillalamarri

    A **production-grade multi-agent AI system** that takes your coding task through a complete DevOps pipeline:

    **4 Specialized AI Agents working in sequence:**
    ✍️ **Code Writer** → 🔍 **Code Reviewer** → 🧪 **Test Writer** → 🚀 **DevOps Agent**

    **Tech Stack:** LangGraph • LangChain • Google Gemini 2.5 Flash • Gradio • Python
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Your Coding Task")

            language = gr.Dropdown(
                choices=["Python", "JavaScript", "Java", "TypeScript", "Go", "C#"],
                label="Programming Language",
                value="Python"
            )

            gr.Markdown("**Quick Load Sample Task:**")
            sample_dropdown = gr.Dropdown(
                choices=list(SAMPLES.keys()),
                label="Sample Tasks",
                value=None
            )

            task_input = gr.Textbox(
                label="Describe Your Coding Task",
                placeholder="e.g. Build a REST API that accepts user data and returns a JSON response...",
                lines=5
            )

            run_btn = gr.Button("🚀 Run Pipeline", variant="primary", size="lg")

            gr.Markdown("### 🔍 Agent Activity Log")
            agent_log = gr.Textbox(
                label="Pipeline Progress",
                lines=8,
                interactive=False,
                placeholder="Agent activity will appear here..."
            )

        with gr.Column(scale=2):
            gr.Markdown("### 📊 Pipeline Results")

            with gr.Tabs():
                with gr.Tab("✍️ Written Code"):
                    written_code_output = gr.Code(
                        label="Code Writer Output",
                        language="python",
                        lines=20
                    )

                with gr.Tab("🔍 Code Review"):
                    review_output = gr.Textbox(
                        label="Code Review Feedback",
                        lines=20,
                        interactive=False
                    )

                with gr.Tab("✅ Revised Code"):
                    revised_code_output = gr.Code(
                        label="Reviewed & Improved Code",
                        language="python",
                        lines=20
                    )

                with gr.Tab("🧪 Unit Tests"):
                    tests_output = gr.Code(
                        label="Generated Unit Tests",
                        language="python",
                        lines=20
                    )

                with gr.Tab("🚀 DevOps Config"):
                    devops_output = gr.Textbox(
                        label="Dockerfile + CI/CD + Deployment Guide",
                        lines=20,
                        interactive=False
                    )

    gr.Markdown("""
    ---
    ### 🏗️ Pipeline Architecture
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

    ---
    Built by **Akshay Pillalamarri** |
    [GitHub](https://github.com/akshaypillalamarri) |
    [LinkedIn](https://www.linkedin.com/in/akshay-pillalamarri) |
    [Portfolio](https://akshaypillalamarri.github.io) |
    [HuggingFace](https://huggingface.co/akshayrinku)
    """)

    sample_dropdown.change(fn=lambda x: SAMPLES.get(x, ""), inputs=sample_dropdown, outputs=task_input)

    run_btn.click(
        fn=run_pipeline,
        inputs=[task_input, language],
        outputs=[written_code_output, review_output, revised_code_output, tests_output, devops_output, agent_log]
    )

demo.launch()
