# Project Jarvis

**A Personal AI Agent Platform powered by Local LLMs, RAG, Memory, and Autonomous Tool Execution.**

Project Jarvis is a modular AI agent platform designed to act as a personal intelligent assistant. It combines local Large Language Model (LLM) inference, Retrieval-Augmented Generation (RAG), persistent memory, and tool execution to help automate tasks, retrieve information, and interact with local systems.

The goal is to build an extensible AI assistant that can understand requests, reason through tasks, use available tools, and execute workflows while maintaining control over local data and infrastructure.

---

## Key Features

- **Local LLM Inference** – Connect to locally hosted language models and inference servers.
- **Retrieval-Augmented Generation (RAG)** – Retrieve relevant information from indexed documents and knowledge bases.
- **Persistent Memory** – Maintain useful context across interactions.
- **Autonomous Tool Execution** – Use tools to perform tasks instead of only generating text.
- **Agent Orchestration** – Coordinate reasoning and tool usage through modular agent workflows.
- **System Automation** – Support interaction with local files, scripts, and system processes through configured tools.
- **Extensible Architecture** – Add new models, tools, integrations, and capabilities as the project evolves.

## Architecture

Project Jarvis is designed around a modular architecture:

```text
                  User
                   |
                   v
             Jarvis Interface
                   |
                   v
            Agent Orchestrator
                   |
          +--------+--------+
          |        |        |
          v        v        v
        LLM      Memory    RAG
       Engine    System   Engine
          |        |        |
          +--------+--------+
                   |
                   v
              Tool Layer
                   |
       +-----------+-----------+
       |           |           |
       v           v           v
   File System   Scripts    External
   Operations   Execution   Services
                   |
                   v
             Task Execution
```

*This is a conceptual architecture; individual components and integrations may evolve during development.*

## Technology Stack

| Component | Technologies |
|---|---|
| Programming Language | Python |
| LLM Inference | llama.cpp, vLLM, local model servers |
| AI Models | Gemma, Qwen, and other compatible models |
| RAG | LangChain, SQLlite, embedding models |
| Agent Framework | Modular agent orchestration |
| API / Backend | Python-based services |
| Frontend | Web-based interface |
| Database / Storage | Configurable local storage and memory |
| Development Tools | Git, GitHub, VS Code, PowerShell |

## Project Structure

The repository is organized into modular components:

```text
Project_Jarvis/
│
├── .github/          # GitHub workflows and automation
├── assets/           # Project assets
├── configs/          # Configuration files
├── deploy/           # Deployment resources
├── desktop/          # Desktop application
├── docs/             # Documentation
├── examples/         # Usage examples
├── frontend/         # Frontend application
├── scripts/          # Utility and automation scripts
├── src/              # Core Python application
├── tests/            # Automated tests
├── rust/             # Rust components
│
├── pyproject.toml    # Python project configuration
├── uv.lock           # Dependency lock file
├── Makefile          # Development commands
├── .gitignore        # Git exclusions
└── README.md         # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.10–3.13
- Git
- [uv](https://docs.astral.sh/uv/)
- A compatible local LLM inference backend, if using local models

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Project_Jarvis
```

### 2. Install Dependencies

```bash
uv sync --group dev
```

### 3. Configure the Environment

Configure the model backend, model path, API endpoints, and other application settings using the project's configuration files.

Keep credentials and private configuration in environment variables or local environment files. Never commit API keys, passwords, or other secrets.

### 4. Run Jarvis

Check the available CLI commands:

```bash
uv run jarvis --help
```

Use the relevant command to start the application or its required services.

## Development

Run automated tests:

```bash
uv run pytest
```

Run code quality checks:

```bash
uv run ruff check .
```

Format Python code:

```bash
uv run ruff format .
```

## Security

Project Jarvis may interact with local files, scripts, processes, and external services through its tools.

- Review tool permissions before enabling execution.
- Avoid exposing sensitive directories or credentials.
- Use environment variables for secrets.
- Validate commands before allowing autonomous execution.
- Restrict access to APIs and services wherever possible.

---
---

## Attribution

Project Jarvis is derived from [OpenJarvis](https://github.com/open-jarvis/OpenJarvis), an open-source AI assistant platform.

The original project is licensed under the Apache License 2.0. This repository retains the upstream license. See [LICENSE](LICENSE).

Project Jarvis includes modifications and extensions maintained by Jeevan Jose.

**Maintainer:** Jeevan Jose
**Status:** Active Development