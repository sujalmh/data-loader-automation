import psycopg2
import pandas as pd
import io
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage

from app.models.model_definition import QualityMetrics, AnalysisResult, FileProcessingResult, ColumnSchema, TableDetails, StructuredIngestionDetails, UnstructuredIngestionDetails, IngestionDetails, FileIngestionResult, IngestionResponse
from app.services.normalization_table import run_normalization

from dotenv import load_dotenv
import os
import re
import mdpd

# --- ENVIRONMENT & LLM SETUP ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# --- DATABASE HELPER FUNCTIONS (POSTGRESQL) ---

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"❌ Error: Could not connect to PostgreSQL database. Please check your connection settings.")
        raise e

def get_db_schema(conn):
    """
    Retrieves the schema of all tables in the public schema of the database.
    This is crucial context for the LLM.
    """
    schema_info = "Database Schema:\n"
    with conn.cursor() as cursor:
        # Get all table names from the 'public' schema
        cursor.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
        """)
        tables = cursor.fetchall()

        if not tables:
            return "" # Return empty if no tables are found

        for table_name_tuple in tables:
            table_name = table_name_tuple[0]
            schema_info += f"\nTable: {table_name}\n"

            # Get column names and types for the current table
            cursor.execute("""
                SELECT column_name, data_type FROM information_schema.columns
                WHERE table_name = %s;
            """, (table_name,))
            columns = cursor.fetchall()
            for col in columns:
                # col structure: (column_name, data_type)
                schema_info += f"  - {col[0]} ({col[1]})\n"
    print("Tables Found:", [t[0] for t in tables])
    return schema_info.strip()

def get_table_schema(conn, table_name):
    """Retrieves the schema for a single table."""
    schema_info = f"Table: {table_name}\n"
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT column_name, data_type FROM information_schema.columns
            WHERE table_name = %s;
        """, (table_name,))
        columns = cursor.fetchall()
        for col in columns:
            schema_info += f"  - {col[0]} ({col[1]})\n"
    return schema_info

def get_table_schema_json(conn, table_name):
    """Retrieves the schema for a single table in JSON format."""
    column_schemas = {}
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT column_name, data_type FROM information_schema.columns
            WHERE table_name = %s;
        """, (table_name,))
        columns = cursor.fetchall()
        for col in columns:
            # col[0] is column_name, col[1] is data_type
            column_schemas[col[0]] = col[1]
    return column_schemas

def execute_query(conn, query, params=None):
    """Executes a given SQL query."""
    with conn.cursor() as cursor:
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
    conn.commit()

def print_table_data(conn, table_name):
    """Prints all rows from a specified table for verification."""
    print(f"\n--- Current Data in '{table_name}' (first 5 rows) ---")
    try:
        # Using f-string for table name is generally safe here as it's internally controlled
        df = pd.read_sql_query(f'SELECT * FROM "{table_name}" LIMIT 5', conn)
        print(df.to_string())
    except Exception as e:
        print(f"Could not read from table '{table_name}': {e}")
    print("------------------------------------------------------\n")


# --- MARKDOWN & DATAFRAME FUNCTIONS ---
def parse_markdown_table(md_table_string):
    return mdpd.from_md(md_table_string)

# --- LLM-POWERED LOGIC (UNCHANGED) ---
# All the functions that interact with the LLM (e.g., get_matching_table_name,
# generate_new_table_details, etc.) do not need to be changed as they are
# database-agnostic. They operate on schema text and dataframes, not direct
# DB connections. I'm omitting them here for brevity but you should keep them
# in your script exactly as they were.

def get_matching_table_name(schema, df_columns, df_sample_rows):
    """
    Uses an LLM to determine if the new data can fit into an existing table.
    """
    print("\nStep 2: Asking LLM to find a matching table...", df_sample_rows)
    schema_description = str(schema) if schema else "The database is empty. There are no tables."

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert database administrator specializing in data normalization. Your task is to determine if new data can be inserted into an existing table, even if it requires restructuring.

    **Database Schema:**
    {schema}

    **New Data to Insert:**
    - **Headers:** {columns}
    - **Sample Rows:**
    {sample_rows}

    **Your Instructions (Follow these strictly):**
    1.  **Analyze New Data Headers:** Look for patterns in the new data headers. Specifically, identify if headers embed metadata like dates (e.g., 'jun_2024'), categories (e.g., 'rural', 'urban'), or other repeating attributes.
        - **Example Pattern:** Headers like 'rural_jun_2024_index', 'urban_jun_2024_index' contain three pieces of information: area ('rural'/'urban'), time ('jun_2024'), and the metric ('index').

    2.  **Compare Semantically:** Compare the *semantic meaning* of the new data with the existing table schemas.
        - **A good match is a table designed to store this *type* of data in a normalized way.** For example, if the new data has a 'rural_jun_2024_index' column, it is a strong match for a table named `price_indices` with columns like `area_type`, `month`, `year`, and `index_value`. The literal names do not have to match.

    3.  **Your Response:**
        - If you find one strong semantic match, respond with ONLY the table name.
        - If you find no strong matches or if the schema is empty, respond with ONLY the word 'None'.
    """),
        ("human", "Based on the schema and new data, what is the matching table name?")
    ])

    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({
        "schema": schema_description,
        "columns": ", ".join(df_columns),
        "sample_rows": df_sample_rows.to_string(index=False)
    })
    
    print(f"LLM Decision: '{result}'")
    return result if result.lower() != 'none' else None

def parse_llm_json(llm_output_str):
    if not isinstance(llm_output_str, str):
        return None
    start_brace = llm_output_str.find('{')
    start_bracket = llm_output_str.find('[')
    
    if start_brace == -1 and start_bracket == -1:
        return None 
    
    if start_brace == -1:
        start_index = start_bracket
    elif start_bracket == -1:
        start_index = start_brace
    else:
        start_index = min(start_brace, start_bracket)

    if llm_output_str[start_index] == '{':
        end_index = llm_output_str.rfind('}')
    else:
        end_index = llm_output_str.rfind(']')

    if end_index == -1:
        return None

    json_str = llm_output_str[start_index : end_index + 1]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

def generate_new_table_details(df, intents, subdomain):
    """
    Uses a single LLM call to generate a table name and a detailed column mapping,
    informed by a data sample. Includes a retry mechanism.
    """
    print("\nStep 3: Asking LLM to generate a complete table schema...")
    
    # --- Improvement 1: Create a data sample ---
    # Get the first 3 rows of the DataFrame as a string for context
    data_sample = df.head(3).to_string()
    df_columns = df.columns.tolist()

    intent_string = ", ".join(intents) if isinstance(intents, list) else intents

    # --- Improvement 2: A single, combined prompt ---
    combined_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert database designer. Your task is to create a complete SQL table schema based on a dataframe's headers and a sample of its data.

        File Intents: {intent_string}
        Dataframe Headers: {columns}
        Subdomain: {subdomain}

        Data Sample (first 3 rows):
        {data_sample}

        ### Instructions:
        1.  Generate a concise, descriptive SQL **table name** in `snake_case`. Follow the naming patterns provided.
        2.  For each dataframe header, generate a corresponding SQL **column name** (in `snake_case`) and infer the most appropriate SQL **data type** (TEXT, REAL, INTEGER).
        3.  **Crucially, use the Data Sample to determine the correct data type.** For example, if a 'month' column contains text like 'March', its type is TEXT, not INTEGER.
        4.  Return a **single JSON object** containing both the 'table_name' and the 'columns' list.

        ### Example Naming Patterns for 'table_name':
        - annual_estimate_gdp_crore
        - city_wise_housing_price_indices
        - exchange_rate_lcy_usd
        - whole_sale_price_index_wpi_calendar_wise

        ### Required JSON Output Format:
        {{
            "table_name": "<generated_table_name>",
            "columns": [
                {{"df_col": "product_name", "sql_col": "product_name", "sql_type": "TEXT"}},
                {{"df_col": "item_price", "sql_col": "price", "sql_type": "REAL"}},
                {{"df_col": "transaction_month", "sql_col": "transaction_month", "sql_type": "TEXT"}}
            ]
        }}
        """),
        ("human", "Generate the complete JSON schema for my table.")
    ])

    # Create a single chain for the combined task
    chain = combined_prompt | llm | StrOutputParser()
    
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            print(f"LLM schema generation: Attempt {attempt + 1}/{max_retries + 1}...")
            
            response_str = chain.invoke({
                "columns": ", ".join(df_columns),
                "intent_string": intent_string,
                "subdomain": subdomain,
                "data_sample": data_sample
            })
            
            result = parse_llm_json(response_str)

            # Validate that the result is a dictionary and both required keys are present
            if isinstance(result, dict) and 'table_name' in result and 'columns' in result:
                print(f"LLM generated schema successfully: {result}")
                return result
            else:
                raise KeyError("LLM response was not a valid dictionary or was missing 'table_name' or 'columns'.")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error on attempt {attempt + 1}: Could not parse LLM response. Error: {e}")
            if attempt < max_retries:
                print("Retrying...")
            else:
                print("All retry attempts have failed.")
                return None
    
    return None

def generate_file_selector_prompt(
    table_name: str,
    headers: list[str],
    sample_rows: list[list[str]],
    intents: str, brief_summary: str, subdomain: str, publishing_authority: str
) -> str:

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert at creating concise and effective prompts for a file selector AI. "
            "Your goal is to generate a 2-3 line description that an AI can use to decide whether "
            "to select a specific table for a user's query. The description must clearly state the "
            "table's purpose, data range (like years), and provide explicit 'DO use' and 'DO NOT use' "
            "conditions based on the rules provided. Follow the style of the examples."
        )),
        ("human", (
            "Generate the prompt for the table '{table_name}'.\n\n"
            "**Data Headers:**\n{headers}\n\n"
            "**Sample Data Rows:**\n{rows}\n\n"
            "**Specific Rules/Intents:**\n{intents}\n\n"
            "**Brief summary:**\n{brief_summary}\n\n"
            "**Subdomain:**\n{subdomain}\n\n"
            "**Publishing Authority:**\n{publishing_authority}\n\n"
            "**Example Style to Follow:**\n"
            "1. cpi_inflation_data: This table contains Consumer Price Index (CPI) and inflation data, categorized by year, month, state, sector (Combined, Rural, Urban), group, and sub-group. It includes inflation trends across categories like food, housing, transport, education, and healthcare. This table covers data for year 2017 to 2025. Do NOT choose this table when \"workers\" or \"labourers\" are mentioned.\n"
            "2. consumer_price_index_cpi_for_agricultural_and_rural_labourers: this table covers data for year 2024. it should be used only when \"agriculture labour\" or \"rural labour\" is mentioned. DO NOT use this file unless \"labour\" is specifically mentioned.\n\n"
            "**Generated Prompt:**"
        ))
    ])

    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({
        "table_name": table_name,
        "headers": ", ".join(headers),
        "rows":"\n".join([", ".join(map(str, row)) for row in sample_rows]),
        "intents":intents,
        "brief_summary":brief_summary,
        "subdomain":subdomain,
        "publishing_authority":publishing_authority
    })
    
    return result

def generate_existing_table_column_map(table_schema, df_columns):
    """Uses an LLM to map DataFrame columns to an existing table's columns."""
    print(f"\nStep 3b: Asking LLM to map columns to existing table...")
    prompt = ChatPromptTemplate.from_template(
        """You are an expert database administrator. Your task is to map columns from a new dataset to an existing SQL table.

Here is the schema of the target SQL table:
{table_schema}

Here are the headers from the new dataset:
{df_columns}

Instructions:
- Create a mapping from the new dataset's headers to the SQL table's columns.
- Return a JSON object with a single key: 'columns'.
- 'columns' should be a LIST of objects, where each object has two keys: 'df_col' (the header from the new data) and 'sql_col' (the corresponding column in the SQL table).
- Only include mappings for columns that have a clear semantic match. If a column from the new data doesn't fit, omit it.
"""
    )
    chain = prompt | llm | StrOutputParser()
    response_str = chain.invoke({
        "table_schema": table_schema,
        "df_columns": ", ".join(df_columns)
    })
    try:
        result = parse_llm_json(response_str)
        if result:
            return result['columns']
        else:
            return []
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing LLM response for column map: {e}")
        return None
# --- END OF LLM-POWERED LOGIC ---

def remove_backslash_except_backslash(text: str) -> str:
    return re.sub(r'\\([^\\])', r'\1', text)

# --- CORE INGESTION LOGIC (MODIFIED FOR POSTGRESQL) ---

def ingest_markdown_table(md_table: str, file_name: str, file_size: int, intents, brief_summary, subdomain, publishing_authority, conn = None) -> FileIngestionResult | TableDetails:
    
    print("=====================================================")
    print("Starting new ingestion process for PostgreSQL...")
    print("=====================================================")

    conn = get_db_connection()
    sql_commands = []

    try:
        print("\nStep 1: Retrieving current database schema...")
        schema = get_db_schema(conn)
        if len(schema.strip()) == 0:
            print("No existing tables found. Will create a new table.")
        
        try:
            df = run_normalization(remove_backslash_except_backslash(md_table))
        except Exception as e:
            print("Error occurred during normalization:", e)
            raise

        file_selector_prompt = None
        if df is None:
            raise Exception("Normalization failed. Aborting.")
        
        target_table = get_matching_table_name(
            schema=schema,
            df_columns=df.columns.tolist(),
            df_sample_rows=df.head(2)
        )
        
        column_map = None

        if target_table:
            table_schema = get_table_schema(conn, target_table)
            column_map = generate_existing_table_column_map(table_schema, df.columns.tolist())
        else:
            schema_details = generate_new_table_details(df, intents, subdomain)
            if not schema_details:
                raise Exception("LLM failed to generate a valid new table schema.")

            if schema_details and 'table_name' in schema_details and 'columns' in schema_details:
                target_table = schema_details['table_name']
                column_map = schema_details['columns']
                
                cols_sql_parts = [f'"{col["sql_col"]}" {col["sql_type"]}' for col in column_map]
                # Use double quotes for table name for case sensitivity and reserved words
                create_sql = f'CREATE TABLE IF NOT EXISTS "{target_table}" ({", ".join(cols_sql_parts)})'
                sql_commands.append(create_sql)
                
                print(f"Executing CREATE TABLE statement: {create_sql}")
                execute_query(conn, create_sql)
                
                file_selector_prompt = generate_file_selector_prompt(
                    table_name=target_table,
                    headers=df.columns.tolist(),
                    sample_rows=df.head(2).values.tolist(),
                    intents=intents,
                    brief_summary=brief_summary,
                    subdomain=subdomain,
                    publishing_authority=publishing_authority
                )
                print(f"Generated Prompt: {file_selector_prompt}")
            else:
                raise Exception("LLM response for new table schema was malformed.")

        if target_table and column_map:
            print(f"\nStep 4: Inserting data into '{target_table}' using explicit column map...")

            sql_cols = [col['sql_col'] for col in column_map]
            df_cols_ordered = [col['df_col'] for col in column_map]
            df_for_insert = df[df_cols_ordered].copy()

            table_schema_json = get_table_schema_json(conn, target_table)
            
            def validate_value(value, sql_type):
                if pd.isna(value) or value == "":
                    return None
                try:
                    sql_type_upper = sql_type.upper()
                    if "INT" in sql_type_upper:
                        return int(value)
                    elif any(t in sql_type_upper for t in ["REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL"]):
                        return float(value)
                    else: # TEXT, VARCHAR, DATE, etc.
                        return str(value)
                except (ValueError, TypeError):
                    return None # Set to NULL if conversion fails

            for i, sql_col in enumerate(sql_cols):
                sql_type = table_schema_json.get(sql_col, "TEXT") # Default to TEXT if not found
                df_for_insert.iloc[:, i] = df_for_insert.iloc[:, i].apply(lambda v: validate_value(v, sql_type))

            columns_str = ', '.join(f'"{c}"' for c in sql_cols)
            # *** KEY CHANGE: Use %s for psycopg2 placeholders instead of ? ***
            placeholders = ', '.join(['%s'] * len(sql_cols))
            insert_sql = f'INSERT INTO "{target_table}" ({columns_str}) VALUES ({placeholders})'
            sql_commands.append(insert_sql)

            rows_to_insert = [tuple(row) for row in df_for_insert.itertuples(index=False)]

            with conn.cursor() as cursor:
                cursor.executemany(insert_sql, rows_to_insert)
            conn.commit()
            print(f"✅ Successfully inserted {len(rows_to_insert)} rows.")

            print_table_data(conn, target_table)

            table_details = get_table_schema_json(conn, target_table)
            
            schema_details_list = [
                ColumnSchema(name=col['sql_col'], type=table_details.get(col['sql_col'], 'UNKNOWN')) 
                for col in column_map
            ]

            return TableDetails(
                tableName=target_table,
                schema_details=schema_details_list,
                rowsInserted=len(rows_to_insert),
                sqlCommands=sql_commands,
                fileSelectorPrompt=file_selector_prompt
            )

        else:
            raise Exception("Could not determine target table or column map.")

    finally:
        if conn:
            conn.close()
            print("PostgreSQL connection closed.")