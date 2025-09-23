import pandas as pd
import os
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# LangChain and Pydantic imports
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, field_validator
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain.output_parsers import OutputFixingParser

# --- Pydantic Models (Unchanged) ---
class NormalizedRow(BaseModel):
    item_description: str = Field(..., description="The primary identifier for the row.")
    region: Optional[str] = Field(None, description="Geographical sub-region.")
    state: Optional[str] = Field(None, description="State or Union Territory.")
    month: Optional[str] = Field(None, description="Month name.")
    year: Optional[int] = Field(None, description="Year.")
    status: Optional[str] = Field(None, description="Status of the data if available.")
    metrics: Dict[str, Optional[float]] = Field(..., description="A dictionary of all metric names (snake_case) to their float values.")

    @field_validator('*', mode='before')
    def clean_nan_and_empty_str(cls, v):
        if isinstance(v, str) and (v.strip() in ['--', '-', 'NA', 'N.A.'] or not v.strip()):
            return None
        return v

class NormalizedTable(BaseModel):
    data: List[NormalizedRow] = Field(..., description="A list of all the normalized data rows from the table.")

    @field_validator('data', mode='before')
    def filter_empty_rows(cls, v):
        """
        Catches and removes any empty dictionary artifacts from the LLM's list output
        before the main validation runs.
        """
        if isinstance(v, list):
            # Filter out any list items that are empty dictionaries
            return [row for row in v if row]
        return v

# --- The core prompt and parser can be defined globally ---
PARSER = PydanticOutputParser(pydantic_object=NormalizedTable)

PROMPT_TEMPLATE = PromptTemplate(
    template="""
    You are an expert data ETL specialist. Your goal is to convert a messy markdown table into a clean, 'tidy' JSON format.

    **Golden Rule: Your response MUST be a valid JSON object and nothing else.** Do not wrap it in markdown backticks. Do not add explanations. Your output should begin with `{{"data":...}}`.

    **Instructions:**
    1.  **Analyze the entire table:** Pay close attention to nested column headers (e.g., 'Rural', 'Urban', 'Combined') and multi-line headers.
    2.  **Unpivot Data:** The input table is "wide". Make it "long". For each state, create a separate JSON object for each combination of region (Rural, Urban, Combined) and month (June, July).
    3.  **Extract Dimensions:**
        - `region`: Extract from the top-level headers ('Rural', 'Urban', 'Combined').
        - `month`, `year`, `status`: Extract from the second-level, multi-line headers (e.g., 'June 24 <br> Index <br> (Final)' implies month='June', year=2024, status='Final').
    4.  **Infer Metrics:** The primary metrics are 'weights' and 'index'. Create a `metrics` dictionary with these `snake_case` keys.
    5.  **Handle Identifiers:** The state/UT name is both the `item_description` and the `state`. For 'All India', set the `state` field to null but keep `item_description` as 'All India'.
    6.  **Data Cleaning:** Convert placeholders like '--', '-', or empty cells to `null`.

    **Example for a Complex Table:**
    *Input Markdown:*
    ```
    | State | Rural | | Urban | |
    |---|---|---|---|---|
    | | Weights | June 24 Index (Final) | Weights | June 24 Index (Final) |
    |---|---|---|---|---|
    | Karnataka | 5.09 | 195.9 | 6.81 | 197.3 |
    | Delhi | -- | 170.5 | 5.64 | 168.7 |
    ```
    
    *Your Expected JSON Output:*
    ```json
    {{
      "data": [
        {{
          "item_description": "Karnataka", "region": "Rural", "state": "Karnataka",
          "month": "June", "year": 2024, "status": "Final",
          "metrics": {{"weights": 5.09, "index": 195.9}}
        }},
        {{
          "item_description": "Karnataka", "region": "Urban", "state": "Karnataka",
          "month": "June", "year": 2024, "status": "Final",
          "metrics": {{"weights": 6.81, "index": 197.3}}
        }},
        {{
          "item_description": "Delhi", "region": "Rural", "state": "Delhi",
          "month": "June", "year": 2024, "status": "Final",
          "metrics": {{"weights": null, "index": 170.5}}
        }},
        {{
          "item_description": "Delhi", "region": "Urban", "state": "Delhi",
          "month": "June", "year": 2024, "status": "Final",
          "metrics": {{"weights": 5.64, "index": 168.7}}
        }}
      ]
    }}
    ```

    **Now, process the following markdown table:**
    {markdown_table}

    {format_instructions}
    """,
    input_variables=["markdown_table"],
    partial_variables={"format_instructions": PARSER.get_format_instructions()}
)
MODEL_NAME = "gemini-2.5-flash"
set_llm_cache(InMemoryCache())
try:
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0, convert_system_message_to_human=True)
except Exception as e:
    print(f"\nERROR: Could not initialize Gemini (Model: {MODEL_NAME}). Check your GOOGLE_API_KEY.")
    print(f"Details: {e}")
    exit()

def run_normalization(md_str: str) -> Optional[pd.DataFrame]:
    """
    Takes a single markdown table string and a pre-initialized LLM, returning a DataFrame.
    Uses an OutputFixingParser for maximum resilience.
    """
    print(f"--- Processing a table... ---")
    
    try:
        # 1. The original parser defines the desired output structure.
        base_parser = PydanticOutputParser(pydantic_object=NormalizedTable)
        
        # 2. The OutputFixingParser wraps the original parser.
        #    If base_parser fails, it intelligently calls the LLM again to fix the output.
        parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

        # The chain now uses the new, self-healing parser
        chain = PROMPT_TEMPLATE | llm | parser
        
        structured_result = chain.invoke({"markdown_table": md_str})

        if not structured_result or not structured_result.data:
            print("Warning: LLM returned no data for this table.")
            return None
            
        data_list = [row.model_dump() for row in structured_result.data]
        df = pd.json_normalize(data_list)
        df.columns = df.columns.str.replace('metrics.', '', regex=False)
        
        id_cols = ['item_description']
        dim_cols = [col for col in ['state', 'region', 'year', 'month', 'status'] if col in df.columns]
        metric_cols = sorted([col for col in df.columns if col not in id_cols and col not in dim_cols])
        
        df.dropna(axis=1, how='all', inplace=True)
        final_cols_order = [c for c in id_cols + dim_cols + metric_cols if c in df.columns]
        
        print("--- Table processed successfully. ---")
        return df[final_cols_order]

    except Exception as e:
        print(f"\nERROR: LLM call failed or output could not be parsed even after attempting a fix. Details: {e}")
        return None

def batch_process_tables(md_strings: List[str], llm: ChatGoogleGenerativeAI, max_workers: int = 5) -> List[pd.DataFrame]:
    """
    Processes a list of markdown table strings in parallel.
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit each markdown string to the executor
        future_to_md = {executor.submit(run_normalization, md_str): md_str for md_str in md_strings}
        
        for future in as_completed(future_to_md):
            try:
                result_df = future.result()
                if result_df is not None:
                    results.append(result_df)
            except Exception as e:
                print(f"An exception occurred while processing a table: {e}")
    return results
