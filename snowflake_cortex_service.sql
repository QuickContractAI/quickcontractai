-- ============================================================================
-- QuickContractAI - Cortex Search Service for the chunked Pile of Law corpus
-- ----------------------------------------------------------------------------
-- Run this AFTER:
--   1. snowflake_setup.sql has created LAWS_CONTRACTS.TEXT.DOCS_CHUNKS_TABLE,
--   2. precheck_before_snowflake_pol.ipynb has finished uploading the ~16.8M
--      chunks into that table.
--
-- The service name (laws_search_service) is what app.py and snowflake_test.py
-- call via SNOWFLAKE.CORTEX.SEARCH_PREVIEW(...).
-- ============================================================================

USE ROLE TRAINING_ROLE;
USE WAREHOUSE DAMG7374;
USE DATABASE LAWS_CONTRACTS;
USE SCHEMA TEXT;

-- ----------------------------------------------------------------------------
-- 1. Build the vector index
--
-- ON CHUNK ............ column the embedding model encodes.
-- ATTRIBUTES .......... columns returned alongside CHUNK in search results;
--                       app.py reads SOURCE / DOC_INDEX / CHUNK_INDEX to pivot
--                       back into DOCS_CHUNKS_TABLE for the full chunk text.
-- EMBEDDING_MODEL ..... snowflake-arctic-embed-l-v2.0 - 1024-dim, English,
--                       handles ~512 input tokens which fits the 1024-char
--                       chunking window from the notebook.
-- TARGET_LAG .......... index is rebuilt at most once a day; the corpus is
--                       static so a tighter lag would just burn credits.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE CORTEX SEARCH SERVICE LAWS_CONTRACTS.TEXT.laws_search_service
    ON CHUNK
    ATTRIBUTES SOURCE, DOC_INDEX, CHUNK_INDEX
    WAREHOUSE = DAMG7374
    TARGET_LAG = '1 day'
    EMBEDDING_MODEL = 'snowflake-arctic-embed-l-v2.0'
    COMMENT = 'Vector index over the chunked Pile of Law corpus; powers RAG retrieval in QuickContractAI.'
    AS (
        SELECT
            SOURCE,
            CHUNK,
            DOC_INDEX,
            CHUNK_INDEX
        FROM LAWS_CONTRACTS.TEXT.DOCS_CHUNKS_TABLE
    );

-- ----------------------------------------------------------------------------
-- 2. Wait for the initial embedding pass to finish
--
-- The first CREATE kicks off a full-corpus embedding job. SHOW CORTEX SEARCH
-- SERVICES reports indexing_state = 'ACTIVE' (and a non-null serving_state)
-- once the index is queryable. For the 16.8M-chunk Pile of Law corpus this
-- typically completes in 2-4 hours on DAMG7374.
-- ----------------------------------------------------------------------------
SHOW CORTEX SEARCH SERVICES IN SCHEMA LAWS_CONTRACTS.TEXT;

DESCRIBE CORTEX SEARCH SERVICE LAWS_CONTRACTS.TEXT.laws_search_service;

-- ----------------------------------------------------------------------------
-- 3. Smoke-test the index from SQL
--
-- Mirrors the call shape used by snowflake_test.py (SEARCH_PREVIEW with a JSON
-- payload). Returns the top-5 chunks for the query plus their attribute
-- columns, which the Python client then uses to fetch the full text.
-- ----------------------------------------------------------------------------
SELECT PARSE_JSON(
    SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
        'LAWS_CONTRACTS.TEXT.laws_search_service',
        '{
            "query": "contract termination",
            "columns": ["SOURCE", "CHUNK", "DOC_INDEX", "CHUNK_INDEX"],
            "limit": 5
        }'
    )
)['results'] AS results;
