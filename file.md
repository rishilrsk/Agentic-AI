

## AGENTIC AI PROJECT
Project Guidelines and Instructions to Students
- Purpose of the Project
As part of the Genrative AI course, students are required to develop and implement an
Agentic AI-based mini project.
The objective of this project is to provide hands-on experience in designing intelligent
systems that can understand a goal, plan tasks, use tools, interact with other agents, retrieve
knowledge, maintain context, and generate an appropriate final response.
The project should demonstrate the practical application of concepts covered in the course.

## 3. Project Requirements
Each project must demonstrate the following Agentic AI characteristics:
## Mandatory Components
- Large Language Model (LLM)
Students may use an appropriate LLM such as:
● OpenAI models
## ● Gemini
## ● Llama
## ● Mistral
## ● Qwen
● Other approved LLMs
● Local LLMs using LM Studio or similar platforms
## 2. Agent
The system must contain at least one functional AI agent.
## 3. Planning / Task Decomposition
The agent should be able to divide a user goal into multiple tasks wherever applicable.
## 4. Tool Usage
At least two tools, APIs, databases, or external resources should be integrated.

## Examples:
● Web search
## ● Python
## ● Calculator
## ● Database
● PDF/document reader
● External API
● Weather API
● File processing
● Knowledge base
## 5. Memory / Context
The system should maintain relevant conversation context or other forms of memory.
## 6. RAG
Where applicable, students should implement Retrieval-Augmented Generation using their
own documents or a suitable knowledge base.
- Multi-Agent Communication
Projects involving multiple agents should demonstrate communication and information
exchange between agents.
## 8. Validation / Error Handling
The system should handle incorrect, incomplete, or unavailable information appropriately.

## 4. Suggested Project Areas
Students may select a project from the following areas or propose another suitable topic with
faculty approval.
S.No. Suggested Project
1 AI Student Academic Assistant
## 2 Research Paper Assistant
## 3 Placement Preparation Agent
4 Resume Analysis and Career Assistant
5 AI Travel Planning Agent
6 Customer Support Multi-Agent System
7 E-Commerce Shopping Assistant
8 AI Personal Finance Assistant
## 9 Healthcare Information Assistant

S.No. Suggested Project
## 10 Smart Agriculture Agent
## 11 College Admission Assistant
## 12 Library Management Agent
13 AI Coding and Debugging Assistant
## 14 Meeting Assistant
## 15 Online Learning Assistant
## 16 Smart City Complaint Management Agent
## 17 College Event Management Agent
18 News Analysis and Summarization Agent
19 Disaster Management Multi-Agent System
## 20 Legal Document Information Assistant
## 5. Important Project Rule
A simple chatbot will NOT be considered an Agentic AI project.
The project should demonstrate an agentic workflow such as:
## User Goal → Planning → Task Decomposition → Tool Use → Knowledge Retrieval →
## Agent Execution → Validation → Final Response
A project that only sends a prompt to an LLM and displays the generated response will not
satisfy the project requirements.
## 6. Recommended System Architecture
Students are encouraged to follow an architecture similar to the following:
## USER
## |
v
## +---------------+
## | User Interface|
## +-------+-------+
## |
v
## +---------------+
## | Planner /     |
## | Supervisor    |
## | Agent         |
## +-------+-------+
## |
## +-----------+-----------+
## |           |           |
v           v           v
## Agent 1     Agent 2     Agent 3
## Research    Analysis    Execution
## |           |           |
## +-----------+-----------+

## |
v
## +---------------+
| Tools / APIs  |
| Python / DB   |
| Search / APIs |
## +-------+-------+
## |
v
## +---------------+
| RAG / Memory  |
| Vector DB     |
## +-------+-------+
## |
v
## +---------------+
## | Validation    |
## | Agent         |
## +-------+-------+
## |
v
## +---------------+
## | Final Response|
## +---------------+
The actual architecture may vary depending on the selected project.
## 7. Recommended Technologies
Students may select technologies based on their project requirements.
## Programming Language
## ● Python – Recommended
● JavaScript / Node.js – Optional
## Agent Frameworks
Students may use:
● LangGraph
● LangChain
● CrewAI
● AutoGen
● Other approved agent frameworks
## LLM
● OpenAI
## ● Gemini
## ● Llama
## ● Mistral
## ● Qwen

● LM Studio/local LLMs
● Other approved models
## Vector Databases
## Examples:
## ● FAISS
● ChromaDB
## ● Pinecone
● Other suitable vector databases
## Interface
## ● Streamlit
## ● Gradio
## ● React
● Web application
● Other suitable interface
Students should clearly mention all technologies and versions used in the project.
## 8. Project Development Stages
Students should complete the project in the following stages.
## Phase 1 – Problem Identification
● Identify a real-world problem.
● Define the problem statement.
● Identify users and requirements.
● Explain why Agentic AI is suitable.
## Phase 2 – Literature Survey
Study existing systems, research papers, frameworks, and applications related to the selected
topic.
Minimum recommended:
5–10 relevant references
## Phase 3 – System Design
## Prepare:
● System architecture
● Agent architecture

## ● Workflow
● Data flow
● Agent communication diagram
● Use case diagram
● Sequence diagram
## Phase 4 – Implementation
## Implement:
● LLM integration
## ● Agent(s)
## ● Planning
● Tool usage
## ● RAG
## ● Memory
● Agent communication
● Error handling
● User interface
## Phase 5 – Testing
Prepare test cases covering:
● Normal inputs
● Multiple-step tasks
● Tool invocation
● Knowledge retrieval
● Invalid inputs
● Error conditions
## Phase 6 – Evaluation
Evaluate the system using appropriate metrics.
## Phase 7 – Documentation
Prepare the final project report according to the format given below.
## Phase 8 – Demonstration
Demonstrate the working prototype and explain the complete architecture.


## 9. Project Report Format

The final report should contain the following sections.
## Cover Page
## Include:
## ● Project Title
● Agentic AI Course Project
## ● Student Name(s)
## ● Roll Number(s)
## ● Branch / Section
## ● Academic Year
## ● Course Name
## ● Faculty Name
## ● Institution Name

## Chapter 1 – Introduction
## 1.1 Background
## 1.2 Problem Statement
## 1.3 Motivation
## 1.4 Objectives
1.5 Scope of the Project

## Chapter 2 – Literature Survey
Discuss existing research and systems related to the selected project.
Include a comparison table:
## Author Year Method Application Advantages Limitations
Students should identify the gap or limitation in existing approaches that motivates their
proposed system.


Chapter 3 – Problem Definition and
## Requirements
## 3.1 Existing System
3.2 Limitations of Existing System
## 3.3 Proposed System
## 3.4 Functional Requirements
3.5 Non-Functional Requirements
## 3.6 Hardware Requirements
## 3.7 Software Requirements
## Chapter 4 – System Design
This chapter should explain the proposed Agentic AI architecture.
## Include:
## 4.1 Overall Architecture
## 4.2 Agent Architecture
4.3 Agent Roles and Responsibilities
4.4 Task Planning and Decomposition
## 4.5 Tool Integration
4.6 RAG Architecture
## 4.7 Memory Architecture
## 4.8 Agent Communication
## 4.9 Data Flow Diagram
## 4.10 Sequence Diagram
## 4.11 Use Case Diagram

## Chapter 5 – Implementation
Explain the actual implementation.
## Include:
5.1 LLM Integration
## 5.2 Prompt Design
## 5.3 Agent Implementation
## 5.4 Planning Mechanism
## 5.5 Tool Integration
5.6 RAG Implementation
## 5.7 Memory Implementation
5.8 Multi-Agent Communication
## 5.9 Error Handling
## 5.10 User Interface
Students should include relevant screenshots and explain each major component.

Chapter 6 – Testing and Results
Students must provide evidence that the system works correctly.
## Test Case Format
## Test Case Input Expected Output Actual Output Status
TC01 User query Correct response Obtained response Pass
TC02 Multi-step request Tasks executed Tasks executed Pass
TC03 Document query Relevant information Retrieved information Pass
TC04 Invalid input Appropriate error Error handled Pass
Students should include screenshots of important outputs.


## 10. Performance Evaluation
Students should select appropriate metrics for their project.
Possible metrics include:
## Metric Description
Task Success Rate Percentage of tasks successfully completed
Response Accuracy Correctness of generated response
Retrieval Accuracy Quality of retrieved information
Response Time Time required to complete a task
Tool Success Rate Successful tool execution
Hallucination Rate Incorrect/generated unsupported information
User Satisfaction User evaluation of the system
Token Usage LLM token consumption
Cost per Task Approximate API cost
Not every metric is mandatory. Students should select metrics relevant to their project.
Chapter 7 – Conclusion and Future Scope
## 7.1 Conclusion
Summarize the work completed and objectives achieved.
## 7.2 Limitations
Clearly mention current limitations.
## 7.3 Future Scope
Discuss possible improvements such as:
● Voice-based agents
● Multimodal agents
● Additional tools
● Advanced memory
● Human-in-the-loop systems
● Local LLM deployment
● Cloud deployment
● Mobile application
● Additional autonomous capabilities


## Chapter 8 – References
Use a consistent citation style.
Students should include:
● Research papers
## ● Books
● Official documentation
● Dataset sources
● APIs
● Framework documentation
All references cited in the report should also appear in the reference list.

## 14. Final Submission Requirements
Each group must submit:
## 1. Project Report
PDF format
## 2. Source Code
Complete and executable source code. With deployment link

