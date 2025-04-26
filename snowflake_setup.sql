-- ============================================================================
-- QuickContractAI - Snowflake provisioning (database / schema / table)
-- ----------------------------------------------------------------------------
-- Run this once before chunking and uploading the Pile of Law corpus from the
-- notebook (precheck_before_snowflake_pol.ipynb). The notebook's Snowpark
-- upload step (`df.write.mode("append").save_as_table("DOCS_CHUNKS_TABLE")`)
-- assumes the database, schema, and table below already exist.
--
-- Connection context expected by app.py / snowflake_test.py:
--   account   = SFEDU02-PDB57018   (Snowflake training account)
--   role      = TRAINING_ROLE
--   warehouse = DAMG7374
--   database  = LAWS_CONTRACTS
--   schema    = TEXT
-- ============================================================================

USE ROLE TRAINING_ROLE;
USE WAREHOUSE DAMG7374;

-- ----------------------------------------------------------------------------
-- 1. Database + schema for the chunked Pile of Law corpus
-- ----------------------------------------------------------------------------
CREATE DATABASE IF NOT EXISTS LAWS_CONTRACTS
    COMMENT = 'Chunked Pile of Law corpus backing the QuickContractAI RAG layer.';

CREATE SCHEMA IF NOT EXISTS LAWS_CONTRACTS.TEXT
    COMMENT = 'Text chunks + Cortex Search Service for retrieval.';

USE DATABASE LAWS_CONTRACTS;
USE SCHEMA TEXT;

-- ----------------------------------------------------------------------------
-- 2. Target table for chunked documents
--
-- Columns match the records produced by the notebook chunking step:
--   { "source": <Pile of Law subset name>,
--     "chunk":  <1024-char chunk with 200-char overlap>,
--     "doc_index":   <int, position of source doc within its batch>,
--     "chunk_index": <int, position of chunk within its source doc> }
--
-- The schema is intentionally lean - text columns are STRING so the
-- snowflake-arctic-embed-l-v2.0 embedding model can ingest them directly.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS LAWS_CONTRACTS.TEXT.DOCS_CHUNKS_TABLE (
    SOURCE       STRING  NOT NULL COMMENT 'Pile of Law subset label (e.g. "Atticus Contracts", "US Code").',
    CHUNK        STRING  NOT NULL COMMENT 'RecursiveCharacterTextSplitter output, ~1024 chars with 200-char overlap.',
    DOC_INDEX    NUMBER  NOT NULL COMMENT 'Position of the source document inside its parsed batch file.',
    CHUNK_INDEX  NUMBER  NOT NULL COMMENT 'Position of the chunk inside its source document.'
);

-- Lookup is by (DOC_INDEX, CHUNK_INDEX) for the full-text fetch in
-- SnowflakeCortexRetriever.get_full_chunk_text(); a search-optimized clustering
-- key keeps the SEARCH_PREVIEW -> full-chunk pivot fast.
ALTER TABLE LAWS_CONTRACTS.TEXT.DOCS_CHUNKS_TABLE
    CLUSTER BY (DOC_INDEX, CHUNK_INDEX);

-- ----------------------------------------------------------------------------
-- 3. Sanity check: confirm the table is empty and ready for the notebook load
-- ----------------------------------------------------------------------------
SELECT COUNT(*) AS row_count FROM LAWS_CONTRACTS.TEXT.DOCS_CHUNKS_TABLE;

-- Expected next step:
--   1. Run cells [1]-[4] of precheck_before_snowflake_pol.ipynb to download,
--      decompress, parse, and chunk the seven Pile of Law subsets.
--   2. Run cells [5]-[6] to upload the ~16.8M chunks into DOCS_CHUNKS_TABLE.
--   3. Run snowflake_cortex_service.sql to build the vector index on top.
