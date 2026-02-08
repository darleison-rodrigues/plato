# 🦫 Project Vision: Platograph

> *A platypus approach to research knowledge graphs*

**Platograph** is a local-first pipeline for independent researchers, product managers, and developers. Like the platypus—a unique hybrid of different animals—Platograph combines PDF processing, vector search, and knowledge graphs into one cohesive research tool.

---

## 1. Core Idea

The tool is **not** a genius AI researcher. It is a **context preparation assistant**. It helps users structure their thinking and workflow around a large collection of documents (PDFs, etc.). The goal is to solve the "I have 20 PDFs and don't know where to start" problem.

The agent becomes a smart research assistant that **preps the work, not does the thinking**.

## 2. Core Workflow

1.  **Ingest:** User dumps a collection of PDFs into a designated folder.
2.  **Analyze:** Lightweight local AI agents analyze the documents to identify types (e.g., research papers, reports, specs), key topics, and high-level relationships.
3.  **Suggest:** The agent proposes potential workflows based on the analysis.
    *   *"I see 5 market research PDFs and 3 competitor analyses. Want me to build a comparison table?"*
    *   *"These look like technical specs. Should I extract all requirements into a markdown template?"*
    *   *"You have 10 customer interview transcripts. Should I extract pain points into categories?"*
4.  **Orchestrate:** User selects a workflow, and the agent orchestrates the extraction, building a structured markdown file from the source documents.
5.  **Synthesize:** The user takes the final, well-structured markdown context to a more powerful tool (ChatGPT, Claude, etc.) for deep work, analysis, and synthesis.

## 3. Technical Stack & Architecture

-   **Orchestration**: LangChain
-   **Local LLM (Agents)**: Ollama (Qwen2.5 1B or Llama 3.2 1B) for lightweight tasks like metadata extraction, pattern recognition, and suggesting templates.
-   **PDF Processing**: PyPDF2 / pdfplumber for robust text extraction.
-   **Context Management**: A custom, simple file-based context manager inspired by the `Contexter` pattern to dynamically manage resources like file handles, temporary folders, and vector DB connections.
-   **Output**: Markdown templates.
-   **CLI**: Typer and Rich for a polished user experience.

### Hardware Constraints

The architecture is designed for low-compute environments.
- **Target Hardware**: i3 10th gen, 8GB RAM.
- **Feasible**: 1B models (~2-3GB RAM), lightweight PDF processing, simple vector embeddings.
- **Out of Scope**: Concurrent processing of many documents, large vector databases, or running 7B+ models.

## 4. Branding & Identity

-   **Name**: Platograph
    -   **Plato** (philosophy, knowledge) + **Graph** (knowledge graph).
    -   A pun on "Platypus Graph".
-   **Metaphor**: The platypus is a unique hybrid (duck bill, beaver tail), just as this tool is a hybrid of PDF processing, vector search, and knowledge graphs.
-   **Logo**: An ASCII platypus for the CLI.

    ```
         ___
      .-'   '-.
     /  o   o  \
    |     <     |
     \  \_____/  /
      '.       .'
        '-----'
       /       \
    ```
