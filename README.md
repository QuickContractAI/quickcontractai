# QuickContractAI

An AI-powered legal document generation and analysis tool that helps create, analyze, and query contract documents with advanced language models and vector search capabilities.

## Installation

### Prerequisites
- Python 3.9+
- Ollama (for local models)
- Snowflake account (for vector search functionality)
- Google API key (for Gemini models)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/QuickContractAI/quickcontractai.git
cd quickcontractai
```

2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

3. Configure your environment variables:
   Edit the `.env` file with your API keys and Snowflake credentials.

4. Start the application:
```bash
source venv/bin/activate
streamlit run app.py
```

#### .env file 

GOOGLE_API_KEY=

SNOWFLAKE_ACCOUNT=

SNOWFLAKE_USER= 

SNOWFLAKE_PASSWORD=

SNOWFLAKE_ROLE=

SNOWFLAKE_WAREHOUSE=

SNOWFLAKE_DATABASE=

SNOWFLAKE_SCHEMA=


## Snowflake provisioning

The RAG layer is backed by a Cortex Search Service over a chunked copy of the
[Pile of Law](https://huggingface.co/datasets/pile-of-law/pile-of-law) corpus.
The first time you stand the project up, run these steps in order:

1. **Create the database, schema, and target table.** From a Snowflake worksheet
   (or `snowsql`) run:
   ```sql
   !source snowflake_setup.sql
   ```
   This creates `LAWS_CONTRACTS.TEXT.DOCS_CHUNKS_TABLE` with the
   `SOURCE / CHUNK / DOC_INDEX / CHUNK_INDEX` schema the app expects.

2. **Download, chunk, and upload the corpus.** Open
   `precheck_before_snowflake_pol.ipynb` and run cells `[1]`–`[6]` end-to-end.
   The notebook:
   - downloads 7 Pile of Law subsets (Atticus Contracts, Resource Contracts,
     ToS, CFPB Consumer Credit, US Code, State Codes, CFR) — ~122K documents,
   - decompresses the `.jsonl.xz` files and normalizes whitespace,
   - splits each document into ~1024-character chunks with 200-char overlap
     via `RecursiveCharacterTextSplitter` (≈16.8M chunks total),
   - uploads the chunks into `DOCS_CHUNKS_TABLE` via Snowpark in parallel
     batches of 10K records.

3. **Build the Cortex Search Service.** Once the table is populated:
   ```sql
   !source snowflake_cortex_service.sql
   ```
   This creates `laws_search_service` on top of `DOCS_CHUNKS_TABLE` using the
   `snowflake-arctic-embed-l-v2.0` embedding model. The initial embedding pass
   over 16.8M chunks takes a few hours on the `DAMG7374` warehouse — poll
   `SHOW CORTEX SEARCH SERVICES` until `indexing_state = ACTIVE` before
   smoke-testing.

4. **Verify retrieval from Python.** Run `python snowflake_test.py` to
   confirm `SEARCH_PREVIEW` returns chunks plus their full-text pivot.

After step 4 succeeds the Streamlit app can be launched normally; the
`SnowflakeCortexRetriever` in `app.py` will talk to `laws_search_service`
directly.

## Overview

QuickContractAI streamlines the contract lifecycle with three main functions:

1. **Contract Generation**: Create customized legal agreements with AI assistance
2. **Document Analysis**: Extract key terms, identify risks, and generate plain-language summaries
3. **Document Q&A**: Ask questions about your contracts and receive accurate answers

## Features

### Contract Generation
- Generate fully customized legal documents from templates
- Choose from multiple contract types (Service Agreements, Employment Agreements, Residential Leases)
- Input refinement with AI suggestions for improved clarity
- Automatic validation for consistency and completeness
- Export to professionally formatted PDF documents
- Jurisdictional customization for all 50 US states
- **CUAD-grounded few-shot prompting** — the draft chain is primed with
  exemplar clauses paraphrased from the Contract Understanding Atticus
  Dataset so the model mimics real commercial-contract structure (see
  `cuad_few_shots.py`)
- **Privacy-preserving cloud step** — before any request to Gemini, party
  names, addresses, and financial figures are swapped on-device for
  opaque tokens (`[PARTY_1]`, `[ADDRESS_1]`, `[AMOUNT_1]`); originals are
  re-injected locally after the draft returns (see `privacy.py`).
  Local refinement and validation chains (Ollama) operate on the real
  values throughout.

### Document Analysis
- Extract key clauses and terms from any contract document
- Identify potential risks with severity ratings (High/Medium/Low)
- Generate plain-language summaries for non-legal professionals
- Analyze custom uploaded documents (PDF, DOCX, TXT)

### Document Q&A
- Query the contract database or your uploaded documents
- Get answers based on intelligent context retrieval
- Filter by document types, jurisdictions, and more
- Maintain conversation history for complex inquiries

## Technical Architecture

QuickContractAI integrates multiple AI models and data systems:

- **Frontend**: Streamlit web interface
- **Large Language Models**:
  - Gemini 2.5 Pro (Google): Main draft generation
  - Gemma 3 (12B): Input refinement
  - Qwen 2.5 Coder (14B): Validation
  - DeepSeek-R1 (14B): Consistency checks
- **Vector Database**: Snowflake with Cortex Vector Search
- **PDF Generation**: FPDF2 library with custom formatting
- **Document Processing**: PyPDF2, docx2txt for document extraction


## Usage

### Contract Generation
1. Select "Contract Generation" from the navigation sidebar
2. Choose a contract type and jurisdiction
3. Fill in the required fields
4. Review AI-suggested refinements
5. Generate and download your contract in PDF format

### Document Analysis
1. Select "Document Analysis" from the navigation sidebar
2. Upload your contract document (PDF, DOCX, or TXT)
3. Choose the document type and jurisdiction
4. Click "Analyze Document" to see extracted clauses, risks, and summaries

### Document Q&A
1. Select "Document Q&A" from the navigation sidebar
2. Type your question about contracts in the chat interface

