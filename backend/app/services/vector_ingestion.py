import os
import base64
import time
import numpy as np
import pandas as pd
from mistralai import Mistral
from dateutil import parser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pymilvus import connections, db, utility, FieldSchema, DataType, Collection, CollectionSchema, MilvusException
from dotenv import load_dotenv
import re
from datetime import datetime
from pymilvus.exceptions import ParamError
from langdetect import detect
from PyPDF2 import PdfReader, PdfWriter

from app.models.model_definition import QualityMetrics, AnalysisResult, FileProcessingResult, ColumnSchema, TableDetails, StructuredIngestionDetails, UnstructuredIngestionDetails, IngestionDetails, FileIngestionResult, IngestionResponse

# ==============================================================================
# 2. Environment Setup and Milvus Connection
# ==============================================================================

# Load environment variables
load_dotenv()

MISTRAL_OCR_KEY = os.getenv("MISTRAL_OCR_KEY")
client = Mistral(api_key=MISTRAL_OCR_KEY)
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", 19530))
MILVUS_DB = os.getenv("MILVUS_DB", "test_db")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "vector_ingestion")

# Connect to Milvus
conn = connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
db_name = MILVUS_DB

if db_name not in db.list_database():
    db.create_database(db_name)
db.using_database(db_name)

# Define schema and collection for Milvus
collection_name = MILVUS_COLLECTION

if not utility.has_collection(collection_name):
    # Define schema
    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
    source_field = FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255)
    page_field = FieldSchema(name="page", dtype=DataType.INT64)
    category_field = FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50)
    embedding_field = FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=768)
    content_field = FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192)
    reference_field = FieldSchema(name="reference", dtype=DataType.VARCHAR, max_length=255)
    date_field = FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=50)
    url_field = FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1000)

    schema = CollectionSchema(
        fields=[id_field, source_field, page_field, category_field,
                embedding_field, content_field, reference_field,
                date_field, url_field]
    )
    collection = Collection(name=collection_name, schema=schema)

else:
    collection = Collection(name=collection_name)

if not collection.indexes:
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200}
    }
    collection.create_index(field_name="embeddings", index_params=index_params)

# Always load before search
collection.load()

# Load Sentence Transformer model
embedding_model_name = 'sentence-transformers/all-mpnet-base-v2'
embedding_model = SentenceTransformer(embedding_model_name, trust_remote_code=True)

# ==============================================================================
# 3. Helper Functions
# ==============================================================================

from datetime import datetime

def to_yyyymm(date_str: str) -> str | None:
    for fmt in ("%b %Y", "%B %Y"):
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y%m")
        except ValueError:
            continue

    try:
        dt = datetime.strptime(date_str, "%Y")
        return dt.strftime("%Y") + "01"
    except ValueError:
        pass

    if "-" in date_str:
        parts = date_str.split("-")
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            start_year = int(parts[0])
            end_year = int(parts[1]) if len(parts[1]) == 4 else int(parts[0][:2] + parts[1])
            return f"{start_year}04"  

    return "202501"

def split_pdf_if_large(file_path, max_size_mb=50, max_pages_per_part=20):
    if os.path.getsize(file_path) / (1024 * 1024) <= max_size_mb:
        return [file_path]
    
    output_dir = os.path.join(os.path.dirname(file_path), 'split_parts')
    os.makedirs(output_dir, exist_ok=True)
    reader = PdfReader(file_path)
    total_pages, split_files = len(reader.pages), []
    
    for i in range(0, total_pages, max_pages_per_part):
        writer = PdfWriter()
        for j in range(i, min(i + max_pages_per_part, total_pages)):
            writer.add_page(reader.pages[j])
        
        part_path = os.path.join(output_dir, f"{os.path.basename(file_path)}_part_{i//max_pages_per_part + 1}.pdf")
        with open(part_path, 'wb') as f:
            writer.write(f)
        split_files.append(part_path)
    return split_files

def filter_english_lines(text):
    english_lines = []
    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped: continue
        if '[SECTION]' in stripped:
            english_lines.append(stripped)
            continue
        try:
            if detect(stripped) == 'en':
                english_lines.append(stripped)
        except:
            continue
    return '\n'.join(english_lines)

def l2_normalize(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

def extract_markdown_from_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            encoded_pdf = base64.b64encode(f.read()).decode("utf-8")
        document = {"type": "document_url", "document_url": f"data:application/pdf;base64,{encoded_pdf}"}
        ocr_response = client.ocr.process(model="mistral-ocr-latest", document=document, include_image_base64=False)
        time.sleep(1)
        pages = ocr_response.pages if hasattr(ocr_response, "pages") else ocr_response
        return [(i + 1, page.markdown) for i, page in enumerate(pages)]
    except Exception as e:
        print(f"Error processing {file_path} with OCR: {e}")
        raise

def content_aware_chunk(text, chunk_size=2500, chunk_overlap=200):
    text = re.sub(r'## (.*?)\n', r'[SECTION] \1\n', text)
    text = re.sub(r'# (.*?)\n', r'[SECTION] \1\n', text)
    chunks = re.split(r'(?=\[SECTION\])', text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def extract_date_from_reference(reference):
    try:
        return parser.parse(reference, fuzzy=True).strftime('%B %Y')
    except (ValueError, TypeError):
        return "Unknown Date"

# ==============================================================================
# 4. Core Processing Logic
# ==============================================================================

def process_and_embed_pdf(file_paths, original_file_name, category, reference, date_str, url):
    """
    Processes PDF file parts, creates chunks and embeddings, and inserts them into Milvus.
    (Refactored for correctness and clarity)
    """
    # Step 1: Extract all text and corresponding page numbers from all PDF parts.
    all_pages = []
    page_offset = 0
    for path in file_paths:
        page_markdown_list = extract_markdown_from_pdf(path)
        if not page_markdown_list:
            raise ValueError(f"Failed to extract markdown from document part {path}")
        
        # Adjust page numbers for multi-part documents
        adjusted_pages = [(page_number + page_offset, markdown) for page_number, markdown in page_markdown_list]
        all_pages.extend(adjusted_pages)
        page_offset += len(page_markdown_list)
    
    if not all_pages:
        raise ValueError("No processable content found in the document.")

    # Step 2: Chunk all the extracted text at once.
    # This is more robust than your original approach.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    final_data_to_embed = []

    for page_number, markdown in all_pages:
        # Split the text of an entire page into smaller chunks
        page_chunks = splitter.split_text(markdown)
        for chunk in page_chunks:
            # Create the combined content that will be embedded
            combined_content = f'Content from {reference}. Page number: {page_number}. Chunk: "{chunk}"'
            
            # Ensure the combined content doesn't exceed the schema's max length
            if len(combined_content) <= 8192:
                final_data_to_embed.append({
                    "content": combined_content,
                    "page": page_number,
                })

    if not final_data_to_embed:
        raise ValueError("Failed to generate any valid chunks for embedding.")

    # Step 3: Create embeddings in a batch (more efficient)
    contents_to_embed = [item["content"] for item in final_data_to_embed]
    embeddings = embedding_model.encode(contents_to_embed, show_progress_bar=False)
    normalized_embeddings = [l2_normalize(emb) for emb in embeddings]
    
    # Step 4: Prepare final, correctly synchronized lists for Milvus insertion.
    num_chunks = len(final_data_to_embed)
    data_to_insert = [
        [original_file_name] * num_chunks,
        [item["page"] for item in final_data_to_embed],
        [category] * num_chunks,
        normalized_embeddings,
        contents_to_embed,
        [reference] * num_chunks,
        [date_str] * num_chunks,
        [url] * num_chunks,
    ]
    
    # Step 5: Insert and FLUSH data.
    try:
        mr = collection.insert(data_to_insert)
        collection.flush()  # <-- CRUCIAL FIX
        print(f'✅ Successfully inserted and flushed {len(mr.primary_keys)} chunks for PDF: {original_file_name}')
        return len(mr.primary_keys)
    except (MilvusException, ParamError) as e:
        print(f'[MILVUS ERROR] for {original_file_name}: {e}')
        raise

# ==============================================================================
# 5. Main Ingestion Function
# ==============================================================================

def ingest_unstructured_file(
    file_path: str,
    file_name: str,
    category: str,
    reference: str,
    url: str,
    fileId: str,
    published_date: str
) -> FileIngestionResult:
    """
    Main function to handle the ingestion of a single unstructured file.
    
    Args:
        file_path: The local path to the file.
        category: The category to assign to the document chunks.
        reference: A reference string for the document (e.g., source name).
        url: The source URL of the document.
        
    Returns:
        A FileIngestionResult object with the status and details of the operation.
    """
    if not os.path.exists(file_path):
        return FileIngestionResult(
            fileName=os.path.basename(file_path),
            fileId=fileId,
            fileSize=0,
            status="failed",
            error="File not found at the specified path."
        )

    file_size = os.path.getsize(file_path)

    try:
        # 1. Extract date from reference metadata
        print(published_date)
        date_str = to_yyyymm(published_date) 
        
        # 2. Split PDF if it's too large for the OCR API
        split_files = split_pdf_if_large(file_path)
        
        # 3. Process the file parts, create embeddings, and insert into Milvus
        chunks_created = process_and_embed_pdf(
            file_paths=split_files,
            original_file_name=file_name,
            category=category,
            reference=reference,
            date_str=date_str,
            url=url
        )
        
        # 4. Create the detailed success response
        ingestion_details = UnstructuredIngestionDetails(
            type="unstructured",
            collection=collection_name,
            chunksCreated=chunks_created,
            embeddingsGenerated=chunks_created,
            chunkingMethod="Content Aware Section Splitting",
            embeddingModel=embedding_model_name
        )
        
        return FileIngestionResult(
            fileName=file_name,
            fileId=fileId,
            fileSize=file_size,
            status="success",
            ingestionDetails=ingestion_details
        )
        
    except Exception as e:
        # 5. Create the failure response
        return FileIngestionResult(
            fileName=file_name,
            fileId=fileId,
            fileSize=file_size,
            status="failed",
            error=f"An unexpected error occurred: {str(e)}"
        )

def collection_search(query, top_k, file_id):
    query_embedding = embedding_model.encode(query)
    query_vector = l2_normalize(np.array(query_embedding)).tolist()

    search_params = {"metric_type": "COSINE", "params": {"ef": 128}}
    print(query, top_k, file_id)

    results = collection.search(
        data=[query_vector],
        anns_field="embeddings",
        param=search_params,
        limit=top_k,
        expr=f'source == "{file_id}"',
        output_fields=["source", "page", "category", "content", "reference", "date", "url"]
    )
    print(results)
    return results
