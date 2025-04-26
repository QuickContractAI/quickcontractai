import os
import dotenv
import snowflake.connector
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()

def get_full_chunk_text(cursor, doc_index, chunk_index):
    """
    Retrieve the full text of a chunk based on its document and chunk indices
    """
    query = f"""
    SELECT CHUNK 
    FROM LAWS_CONTRACTS.TEXT.DOCS_CHUNKS_TABLE
    WHERE DOC_INDEX = {doc_index} AND CHUNK_INDEX = {chunk_index}
    """
    
    cursor.execute(query)
    result = cursor.fetchone()
    
    if result:
        return result[0]
    else:
        return "Full text not found"

def main():
    print("Starting Snowflake Cortex search service test...")
    
    # Connection parameters
    connection_parameters = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT", "SFEDU02-PDB57018"),
        "user": os.getenv("SNOWFLAKE_USER", "CAT"),
        "password": os.getenv("SNOWFLAKE_PASSWORD", ""),  
        "role": os.getenv("SNOWFLAKE_ROLE", "TRAINING_ROLE"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "DAMG7374"),
        "database": os.getenv("SNOWFLAKE_DATABASE", "LAWS_CONTRACTS"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA", "TEXT")
    }
    
    print("Connection parameters loaded (excluding password)")
    
    try:
        print("Attempting to connect to Snowflake...")
        conn = snowflake.connector.connect(**connection_parameters)
        print("Connection successful!")
        
        cursor = conn.cursor()
        
        # Define search parameters
        search_query = "contract termination"  # Replace with your desired search query
        
        print(f"Querying Cortex search service using SEARCH_PREVIEW function...")
        
        # Using the exact format from the successful query
        query = f"""
        SELECT PARSE_JSON(
          SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
            'LAWS_CONTRACTS.TEXT.laws_search_service',
            '{{
              "query": "{search_query}", 
              "columns": ["SOURCE", "CHUNK", "DOC_INDEX", "CHUNK_INDEX"],
              "limit": 5
            }}'
          )
        )['results'] AS results;
        """
        
        cursor.execute(query)
        
        # Fetch results
        results = cursor.fetchall()
        print(f"Search results count: {len(results)}")
        
        # Parse and display the results
        for i, row in enumerate(results):
            print(f"\nResult {i+1}:")
            # The row contains a JSON string that we need to parse
            result_json = row[0]
            if isinstance(result_json, str):
                result_data = json.loads(result_json)
            else:
                result_data = result_json
                
            # Print the result in a readable format
            if isinstance(result_data, list):
                for j, item in enumerate(result_data):
                    print(f"  Item {j+1}:")
                    
                    # Store doc_index and chunk_index for retrieving full text
                    doc_index = item.get("DOC_INDEX")
                    chunk_index = item.get("CHUNK_INDEX")
                    
                    # Print the metadata
                    for key, value in item.items():
                        if key != "CHUNK":  # Skip the truncated chunk
                            print(f"    {key}: {value}")
                    
                    # Retrieve and print the full chunk text
                    if doc_index is not None and chunk_index is not None:
                        try:
                            full_chunk_text = get_full_chunk_text(cursor, doc_index, chunk_index)
                            print(f"    FULL_CHUNK: {full_chunk_text}")
                        except Exception as e:
                            print(f"    Error retrieving full chunk: {str(e)}")
                            print(f"    TRUNCATED_CHUNK: {item.get('CHUNK', 'N/A')}")
                    else:
                        print(f"    TRUNCATED_CHUNK: {item.get('CHUNK', 'N/A')}")
            else:
                for key, value in result_data.items():
                    # Truncate long values for readability
                    if isinstance(value, str) and len(value) > 200:
                        print(f"  {key}: {value[:200]}...")
                    else:
                        print(f"  {key}: {value}")
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        logger.error(f"Exception details: {e}", exc_info=True)
        
if __name__ == "__main__":
    main()