# QuickContractAI

> AI-powered legal document generation, analysis, and Q&A — powered by Gemini, local LLMs via Ollama, and Snowflake vector search.

## Overview

QuickContractAI streamlines the contract lifecycle with three main capabilities:

1. **Contract Generation** — Create fully customized legal agreements from templates using AI assistance, validated for consistency, and exported as PDF
2. **Document Analysis** — Extract key clauses, identify risks (with High/Medium/Low severity ratings), and generate plain-language summaries from uploaded contracts
3. **Document Q&A** — Query the contract database or your own uploaded documents using natural language, with context retrieved via Snowflake Cortex Vector Search

The system uses a multi-model architecture: Gemini 2.5 Pro for drafting, local models (Gemma 3, Qwen 2.5 Coder, DeepSeek-R1) for refinement and validation, and Snowflake for vector storage and retrieval.

## Features

### Contract Generation
- Choose from Service Agreements, Employment Agreements, and Residential Leases
- AI-driven input refinement with suggestions for improved clarity
- Automatic consistency and completeness validation
- Jurisdictional customization for all 50 US states
- Export to professionally formatted PDF documents

### Document Analysis
- Upload PDF, DOCX, or TXT contract files
- Extract key clauses and terms automatically
- Risk identification with severity ratings
- Plain-language summaries for non-legal professionals

### Document Q&A
- Ask natural-language questions about any contract
- Context-aware answers backed by vector search
- Conversation history maintained across queries

## Requirements

| Tool | Version |
|------|---------|
| Python | 3.9+ |
| Ollama | latest |
| Snowflake account | for vector search |
| Google API key | for Gemini models |

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/QuickContractAI/quickcontractai.git
cd quickcontractai
```

### 2. Run the automated setup script

```bash
chmod +x setup.sh
./setup.sh
```

The script will:
- Create a Python virtual environment
- Install all Python dependencies
- Create a `.env` file from the template
- Pull required Ollama models (TinyLlama 1.1B, ShieldGemma 2B, Phi4 14B)

> Note: Pulling large Ollama models requires several GB of disk space and may take time on first run.

### 3. Configure environment variables

Edit the `.env` file created by the setup script:

```env
GOOGLE_API_KEY=your_google_api_key_here

SNOWFLAKE_ACCOUNT=your_account_identifier
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ROLE=your_role
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema
```

### 4. Start the application

```bash
source venv/bin/activate
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

## Usage

### Contract Generation
1. Select **Contract Generation** from the sidebar
2. Choose a contract type and jurisdiction
3. Fill in the required fields
4. Review AI-suggested refinements
5. Generate and download the contract as PDF

### Document Analysis
1. Select **Document Analysis** from the sidebar
2. Upload a contract document (PDF, DOCX, or TXT)
3. Choose the document type and jurisdiction
4. Click **Analyze Document** to view extracted clauses, risks, and summary

### Document Q&A
1. Select **Document Q&A** from the sidebar
2. Type your question in the chat interface
3. Receive answers grounded in the contract database or your uploaded document

## Technical Architecture

```
Streamlit Frontend
       |
       +---> Contract Generation ---> Gemini 2.5 Pro (draft)
       |                          ---> Gemma 3 12B (refinement)
       |                          ---> Qwen 2.5 Coder 14B (validation)
       |                          ---> DeepSeek-R1 14B (consistency)
       |                          ---> FPDF2 (PDF export)
       |
       +---> Document Analysis ----> PyPDF2 / docx2txt (extraction)
       |                          ---> Gemini (analysis)
       |
       +---> Document Q&A ---------> Snowflake Cortex Vector Search
                                  ---> Gemini (answer generation)
```

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| LLM (cloud) | Gemini 2.5 Pro via Google Generative AI |
| LLM (local) | Ollama (TinyLlama, Phi4, Gemma 3, Qwen, DeepSeek-R1) |
| Vector database | Snowflake with Cortex Vector Search |
| PDF generation | FPDF2 |
| Document parsing | PyPDF2, docx2txt |

## Project Structure

```
INFO7374/
├── app.py                        # Streamlit entry point and navigation
├── main.py                       # Core application logic
├── contract_analysis.py          # Document analysis pipeline
├── document_upload_analysis.py   # Document upload and parsing
├── snowflake_test.py             # Snowflake connectivity tests
├── precheck_before_snowflake_pol.ipynb  # Pre-deployment checks
├── requirements.txt              # Python dependencies
└── setup.sh                      # Automated setup script
```

## Course

INFO 7374 — Northeastern University, Spring 2025

## License

MIT
