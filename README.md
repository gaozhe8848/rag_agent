## 🏗️ System Architecture

This project implements a Retrieval-Augmented Generation (RAG) agent designed to act as an IT Service Management expert. It uses a **single-agent, single-tool** architecture optimized for distinct user workflows (Summary, Root Cause Analysis, and Status Checks).

The system is built using **LangChain** for orchestration, **LangAgent** for the agent runtime loop, and **OpenAI** for embeddings and generation.

### Component Breakdown

#### 1. Ingestion Pipeline (Startup Phase)
* **Raw Data Sources:**
    * `remedy_data.json`: Contains historical work logs for multiple support tickets.
    * `remedy_guide.json`: A structured "rulebook" defining how to interpret logs for root cause, status, and contacts.
* **Loader & Metadata Extractor:** Uses LangChain's `JSONLoader` to read the files. Critical metadata (Incident ID, Submitter, Log Type, Date) is extracted and attached to each chunk.
* **Embeddings & Vector Store:** The text chunks are converted into vector embeddings using OpenAI's `text-embedding-3-large` model and stored in an in-memory vector database for semantic retrieval.

#### 2. Agent Runtime (Query Phase)
* **LangGraph ReAct Agent:** The core orchestrator that operates in a "Reason and Act" loop. It receives the user query and decides if it needs to use tools to answer it.
* **System Prompt:** A carefully crafted set of instructions that tells the agent how to behave (as an ITSM expert) and how to apply the rules found in the guide based on the user's requested workflow (e.g., "If asked for Root Cause, filter strictly for 'Resolution' logs").
* **Tool: `retrieve_ticket_history`:** The single tool available to the agent. It performs a **filtered similarity search** on the vector store. It ensures that the agent only receives logs relating to the specific requested Incident ID, alongside the general analysis guide.
* **LLM (e.g. gpt-5.1):** The underlying language model that processes the retrieved context and generates the final, synthesized answer according to the system prompt's rules.