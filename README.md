# University_Knowledge_Assistant_UoP_Policies
University Knowledge Assistant — University of Peshawar Policies

A Retrieval-Augmented Generation (RAG) application built with Streamlit, FAISS,
Sentence Transformers, and the Groq API. It retrieves relevant pre-indexed text
chunks and asks a Groq-hosted language model to answer with source traceability.

Important: no original PDFs are required at runtime

The application does not upload, download, or process the original PDFs. It
loads the precomputed FAISS index and aligned chunk metadata created by the
separate Google Colab ingestion workflow.

chunks.jsonl contains extracted text chunks as well as metadata. It is not
an embeddings-only file. The FAISS index stores vectors; the JSONL file stores
the text that the model needs as context and the metadata needed for citations.

Repository structure

University_Knowledge_Assistant_UoP_Policies/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── vector_store/
    ├── index.faiss
    ├── chunks.jsonl
    ├── documents_manifest.json
    ├── config.json
    └── package_versions.json

Do not add original PDFs, Google Drive credentials, or API keys to this repository.
Only include documents/text chunks that you have permission to store and publish.
If the extracted policy text is not intended for public access, keep the repository
private and review the hosting provider's access model.

How it works

Load vector_store/index.faiss and vector_store/chunks.jsonl.

Embed the student's question with the same embedding model used during ingestion.

Search the normalized FAISS vectors using inner product (cosine similarity).

Pass the top retrieved text chunks to Groq.

Show the answer and source metadata: document title, page, chunk ID, and similarity.

Document embeddings are not regenerated at startup. Only the question is embedded
for each search.

Metadata and traceability

The ingestion workflow should store one JSON object per line in chunks.jsonl,
with at least these fields:

chunk_id

document_id

document_title

file_name

page_number

chunk_index

text

document_sha256

source_folder_url

official_status

Additional fields such as department, program, degree level, session/year, issue
date, effective date, and section title may be included. Populate them only when
they can be verified. The app labels the original source title/page/chunk and
shows a similarity score; similarity is not a measure of official authority.

The index and metadata must be generated from the same ingestion run and must have
the same number of records/vectors. The app checks this when loading.

Configure Groq

This app defaults to the model ID openai/gpt-oss-120b, as requested. Model access
depends on current Groq availability and your account's access. If that model is
unavailable, enter a currently supported model ID in the sidebar.

Streamlit Community Cloud

Push this repository to GitHub.

Open Streamlit Community Cloud and create an app from this repository.

Select app.py as the entrypoint.

In app settings, open Secrets and add:

GROQ_API_KEY = "your_groq_api_key"

Never commit your real API key to GitHub.

Local run

Use Python 3.10 or newer, install dependencies, and set the API key as an environment
variable. Then run Streamlit from the repository root.

pip install -r requirements.txt

Windows PowerShell:

$env:GROQ_API_KEY="your_groq_api_key"
streamlit run app.py

macOS/Linux:

export GROQ_API_KEY="your_groq_api_key"
streamlit run app.py

Deployment notes

Keep app.py and requirements.txt at the repository root.

Keep all five vector-store artifacts in vector_store/.

Use the exact same embedding model that created the FAISS index.

If index.faiss and chunks.jsonl are mismatched, rerun the ingestion workflow and upload both artifacts from that run.

A large vector store may exceed GitHub's file-size limits. For larger stores, place the artifacts in controlled object storage and download them at startup, or use Git LFS where appropriate.

Avoid claiming that retrieved text is current official policy unless you have verified its source and effective date.

Limitations

This is a student project/reference assistant, not an official University of Peshawar service.
The answer can only be as reliable as the documents indexed. Check current notifications,
prospectuses, regulations, and department-specific rules before acting on deadlines,
fees, attendance thresholds, penalties, or exceptions.
