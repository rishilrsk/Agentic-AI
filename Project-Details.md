# AI Project Development Prompt — Agentic Data Analyst

Act as my senior AI engineer, software architect, Python mentor, and project development partner. Help me design, implement, test, and document a complete Agentic AI project, working with me one phase at a time.

I am a final-year Computer Science and Engineering student. I want to build a functional, well-documented project that meets my Agentic AI course requirements and is suitable for a college demonstration, viva, GitHub portfolio, and resume.

I need practical implementation help, not just project suggestions or theoretical explanations.

## 1. Project Title

**Agentic Data Analyst: An Intelligent Multi-Step Data Analysis and Insight Generation System**

## 2. Project Objective

Build an AI-powered data analysis application that enables users to upload CSV or Excel datasets and ask analytical questions in natural language.

The system should be capable of:
- Understanding the user's analytical goal.
- Inspecting the dataset's schema, columns, data types, and quality.
- Breaking complex questions into smaller analytical tasks.
- Generating and executing appropriate SQL queries.
- Using actual query results rather than inventing numerical answers.
- Detecting query errors and attempting safe corrections.
- Generating charts, tables, and evidence-based explanations.
- Remembering relevant previous questions and analysis sessions.
- Retrieving relevant dataset documentation and field definitions when needed.
- Validating the final results before presenting them.

Example user request:

"Analyze this sales dataset. Identify the top five products by revenue, compare monthly revenue, find the city with the largest sales decline, and present the results in charts."

The system must plan and execute the required analytical steps rather than answer using a single LLM prompt.

## 3. First, Read My Course Guidelines

I will attach my official Agentic AI project guidelines PDF.

Read the document before finalizing the design. Extract its actual mandatory requirements, recommended technologies, report format, evaluation metrics, and submission requirements.

Ensure that the final implementation addresses the following requirements specified for the course:

1. LLM integration.
2. At least one functional AI agent.
3. Planning and task decomposition.
4. Integration of at least two tools, APIs, databases, or external resources.
5. Memory or contextual continuity.
6. RAG where applicable.
7. Agent communication when using multiple agents.
8. Validation and error handling.
9. Testing and performance evaluation.
10. Project report, source code, and demonstration readiness.

Do not treat a simple natural-language-to-SQL chatbot as sufficient. Demonstrate an actual workflow involving planning, tool execution, result verification, and final response generation.

Follow the exact report structure in the attached PDF. If the PDF is unavailable, tell me what is missing and proceed using the project description in this prompt.

## 4. Technology Stack

Prefer this free, local-first stack unless a technical limitation justifies a change.

- Programming language: Python.
- Frontend: Streamlit.
- Agent orchestration: LangGraph.
- Language model: Ollama with a suitable locally runnable model.
- Data processing: Pandas.
- Analytical SQL engine: DuckDB.
- Visualizations: Plotly.
- Persistent application memory: SQLite.
- Optional vector database for RAG: ChromaDB.
- Version control: Git and GitHub.

Use stable, mutually compatible package versions. Check compatibility before recommending installations.

I do not want any mandatory paid subscriptions, premium models, paid APIs, or paid database services.

The application should use a local LLM by default. Do not silently replace it with a cloud API if it performs poorly or reaches a limit. Explain the limitation and ask me before proposing any cloud alternative.

If my computer cannot run the recommended model efficiently, suggest a smaller model based on my actual hardware specifications.

## 5. Required Agent Architecture

Design a supervisor-based workflow using LangGraph. Keep the implementation understandable and avoid unnecessary complexity.

Proposed components:

### A. Supervisor / Planner Agent
- Understand the analytical request.
- Determine whether one query or multiple steps are required.
- Create a structured execution plan.
- Decide which tools to invoke.
- Track completed and pending tasks.

### B. Dataset Profiling Agent
- Inspect columns and data types.
- Identify missing values and duplicate records.
- Determine which fields are suitable for the requested analysis.
- Build a reliable schema description.

### C. SQL Analysis Agent
- Convert the analysis plan into SQL.
- Use DuckDB to execute queries against the uploaded dataset.
- Return structured results with the executed query.
- Never invent or estimate query results.

### D. Validation and Recovery Agent
- Validate generated SQL before execution.
- Detect syntax errors, missing columns, incompatible data types, and empty results.
- Verify calculations where possible.
- Attempt a bounded number of corrections when appropriate.
- Report unresolved errors honestly.

### E. Insight and Visualization Agent
- Convert verified results into understandable findings.
- Generate appropriate charts and tables.
- Explain important trends using the actual output.
- Distinguish observed facts from hypotheses or potential causes.

These may be implemented as cooperating LangGraph nodes rather than separate model deployments. Do not require a separate LLM for every agent.

Use structured state to pass the goal, plan, schema, queries, outputs, errors, validation status, and final response between components.

Only describe the application as multi-agent to the extent that the implementation actually supports those roles and their communication.

## 6. Tool Integration

Implement useful tools such as:

1. File upload and CSV/Excel parser.
2. Dataset profiling and schema inspection.
3. DuckDB SQL execution.
4. Query result validation.
5. Chart generation.
6. Dataset documentation retrieval.
7. SQLite-based analysis history and memory.

Each tool should have a clearly defined input, output, and error-handling mechanism.

Implement RAG only where it adds value, such as retrieving data dictionary entries, business definitions, dataset documentation, and relevant analysis notes.

## 7. Security and Reliability

Treat security as a required feature.

- Permit read-only analytical SQL only.
- Validate SQL before executing it.
- Restrict queries to the uploaded dataset.
- Block database modification, file-writing, installation, and external-access commands.
- Limit query execution time and returned row counts.
- Validate uploaded file types and sizes.
- Handle malformed files and missing columns gracefully.
- Do not execute arbitrary model-generated Python code.
- Never fabricate a successful test result.
- Ask for permission before destructive file operations or major project-wide changes.
- Do not store passwords or API keys in source code.

Use reliable data processing for calculations and use the LLM primarily for interpretation, planning, and query generation.

## 8. Development Phases

Build the project in the following order.

### Phase 0 — Environment and feasibility
Inspect my operating system, Python version, available RAM, storage, and installed tools. Help me verify that the selected local model is suitable.

### Phase 1 — Project blueprint
Finalize the problem statement, objectives, scope, system architecture, agent responsibilities, data flow, use cases, and acceptance criteria. Identify the project's contribution beyond a basic NL2SQL tool.

### Phase 2 — Environment setup and scaffold
Create the project directory, virtual environment, dependency files, configuration, README, and basic application entry point. Verify that the empty application starts successfully.

### Phase 3 — Dataset ingestion and profiling
Implement CSV/Excel upload, schema inspection, data types, missing-value reports, and dataset summaries. Test with a small sample dataset.

### Phase 4 — SQL analysis engine
Integrate DuckDB. Implement schema-aware SQL generation support, safe query validation, query execution, structured result handling, and error reporting.

### Phase 5 — Local LLM integration
Connect the application to Ollama. Verify model availability, prompts, structured outputs, and failures. Do not assume the model supports reliable tool calling until tested.

### Phase 6 — Agentic workflow
Implement the LangGraph supervisor and analytical workflow, including planning, task decomposition, tool invocation, shared state, validation, and bounded recovery.

### Phase 7 — Insights and visualization
Generate charts and explain the results. Ensure that the explanation agrees with the verified query output.

### Phase 8 — Memory and RAG
Add persistent question history and conversational context. Implement retrieval over relevant dataset documentation and field definitions where applicable.

### Phase 9 — Frontend and user experience
Build an intuitive dashboard with dataset upload, dataset preview, chat interface, execution status, SQL inspection, results, charts, and analysis history.

### Phase 10 — Testing and evaluation
Test normal requests, multi-step requests, invalid inputs, query failures, missing columns, empty results, ambiguous questions, and unavailable models.

Evaluate relevant metrics such as:
- Task success rate.
- SQL execution success rate.
- Query result correctness.
- Tool success rate.
- Recovery success rate.
- Response time.
- Retrieval quality where applicable.
- Approximate local compute and resource usage.

Record real test results. Do not claim accuracy percentages without a defined evaluation method and actual measurements.

### Phase 11 — Documentation and submission
Prepare the complete report according to the supplied PDF, architecture diagrams, workflow diagrams, use case diagram, sequence diagram, screenshots, test-case table, references, setup instructions, and demonstration script.

Provide a deployment option only if it is practical within the zero-cost constraint. A working local demonstration is the primary requirement.

## 9. Rules for Every Development Phase

For every phase, follow this format:

1. **Objective:** Explain what we are building and why.
2. **Files:** Show the exact directory tree and the files to create or modify.
3. **Implementation:** Provide complete, runnable code for every new or modified file. Do not use placeholders such as "implement the remaining logic here."
4. **Installation:** Provide exact commands and package requirements.
5. **Execution:** Explain how to start the application or run the relevant script.
6. **Verification:** Provide tests or manual steps with expected results.
7. **Troubleshooting:** Explain likely errors and how to resolve them.
8. **Completion checklist:** Identify what has been completed and what remains.

Maintain a clear project checklist and update it after each phase. Keep the README and configuration synchronized with the implementation.

When modifying an existing file, inspect its current content first and provide the complete revised file. Avoid unrelated rewrites.

Explain technical decisions in beginner-friendly English, but keep the implementation at a standard appropriate for a final-year CSE project.

If you have access to my project workspace, create and test files there. Otherwise, provide file-by-file code and exact instructions for implementing it locally.

Do not dump the entire application in one response.

## 10. How to Begin

Your first response must:

1. Confirm your understanding of the project.
2. Summarize the requirements extracted from my attached guidelines.
3. Present the finalized proposed architecture and phase roadmap.
4. List the tools and software that must be installed.
5. Identify any hardware limitations that may affect the local LLM.
6. Establish measurable acceptance criteria for the MVP.

Do not start implementing the entire project immediately.

Begin with Phase 0 and the Phase 1 blueprint. After presenting those deliverables, stop and wait for me to say **"Continue to Phase 2"** before proceeding.

Continue this process for every subsequent phase. Preserve project context and decisions throughout the conversation.

My final objective is a genuinely functioning, secure, locally runnable Agentic Data Analyst that satisfies the course guidelines and that I can explain confidently during my project demonstration and viva.

