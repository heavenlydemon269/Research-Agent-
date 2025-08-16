# Research-Agent-

graph TD
    subgraph "USER INTERFACE (Frontend)"
        A[Streamlit Web App]
        User(User) -- "1. Inputs URL & Query" --> A
    end

    subgraph "BACKEND LOGIC (LangChain Orchestration)"
        B[Document Loaders <br/>(Web, PDF, YouTube)]
        C[Text Splitter <br/>(RecursiveCharacterTextSplitter)]
        D{FAISS Vector Store <br/>(In-Memory Knowledge Base)}
        E[Similarity Search]
        F[QA Chain <br/>(load_qa_chain)]

        A -- "2. Sends URL" --> B
        B -- "3. Raw Text" --> C
        C -- "4. Text Chunks" --> D
        A -- "5. Sends User Query" --> E
        E -- "6. Performs Search on Query" --> D
        D -- "7. Returns Relevant Chunks" --> F
        E -- "User Query" --> F
    end

    subgraph "EXTERNAL SERVICES (APIs)"
        G[OpenAI Embeddings API]
        H[OpenAI LLM API <br/>(e.g., GPT-4)]

        C -- "Generates Embeddings For Chunks" --> G
        G -- "Embeddings" --> D
        F -- "8. Sends Query + Chunks to LLM" --> H
        H -- "9. Generates Final Answer" --> F
    end

    F -- "10. Final Answer" --> A

    style User fill:#D6EAF8,stroke:#333,stroke-width:2px
    style A fill:#E8DAEF,stroke:#333,stroke-width:2px
    style D fill:#D5F5E3,stroke:#333,stroke-width:4px
    style H fill:#FCF3CF,stroke:#333,stroke-width:2px
