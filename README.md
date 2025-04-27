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
git clone https://github.com/yourusername/quickcontractai.git
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

SNOWFLAKE_

SNOWFLAKE_PASSWORD=

SNOWFLAKE_ROLE=

SNOWFLAKE_WAREHOUSE=

SNOWFLAKE_DATABASE=

SNOWFLAKE_SCHEMA=


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

