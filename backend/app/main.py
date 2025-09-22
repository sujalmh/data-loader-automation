# main.py
import os
import shutil
from typing import List

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.services.process_file import process_pdf, extract_markdown_from_file, extract_markdown_from_excel_csv
from app.services.vector_ingestion import ingest_unstructured_file, collection_search

from app.models.model_definition import QualityMetrics, AnalysisResult, FileProcessingResult, FileIngestionResult, IngestionResponse, DownloadResult, UrlListRequest, FileMetadata, DownloadSuccess, DownloadError, DownloadDuplicate, StructuredIngestionDetails#, ProcessingResult, SearchResponse, SearchResult, SearchResultEntity, SearchRequest
from app.services.table_agent import ingest_markdown_table
from app.services.file_handler import extract_markdown_tables

import app.models.db as db

import hashlib
import asyncio
from pathlib import Path
from typing import List, Dict, Tuple, AsyncGenerator
from urllib.parse import urlparse, unquote
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from starlette.responses import StreamingResponse, FileResponse
import tempfile
import json
import aiofiles
import uuid
import base64


UPLOAD_DIRECTORY = Path("uploaded_files")
MARKDOWN_DIRECTORY = "markdown_output"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for the application.
    This replaces the deprecated on_event("startup") and on_event("shutdown").
    """
    # --- Startup Event ---
    UPLOAD_DIRECTORY.mkdir(exist_ok=True)
    await db.connect_db()
    await db.init_db()

    yield
    # --- Shutdown Event ---
    await db.disconnect_db() 
    print("Application shutdown complete.")

# --- App Initialization ---
# Create a FastAPI app instance
app = FastAPI(
    title="File Ingestion API",
    description="An API to receive and store user-uploaded files.",
    lifespan=lifespan
)

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://34.41.241.77:8073",
    "http://localhost:8073",
    "http://0.0.0.0:3000",
    "http://100.104.12.231:8073"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def calculate_sha256(content: bytes) -> str:
    """Calculates the SHA-256 hash of the file content."""
    sha256_hash = hashlib.sha256()
    sha256_hash.update(content)
    return sha256_hash.hexdigest()

# --- API Endpoints ---
@app.post("/upload-files/", summary="Upload and Store Files Asynchronously")
async def upload_and_store_files(files: List[UploadFile] = File(...)):
    """
    Handles file uploads asynchronously, saving them to the server with a unique ID.

    This endpoint generates a UUID for each file to prevent filename collisions
    and returns detailed metadata, including the unique ID, for each
    successfully stored file.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were sent.")

    response_data = []
    duplicates_data = []

    for file in files:
        try:
            file_id = str(uuid.uuid4())
            original_filename = file.filename or "unknown"
            suffix = Path(original_filename).suffix
            unique_filename = f"{file_id}{suffix}"
            file_path = UPLOAD_DIRECTORY / unique_filename

            content = await file.read()
            async with aiofiles.open(file_path, "wb") as buffer:
                await buffer.write(content)

            file_hash = calculate_sha256(content)
            existing_file = await db.check_duplicate(file_hash)
            if existing_file:
                duplicates_data.append({
                    "name": file.filename or "unknown",
                    "message": "File is a duplicate of an existing record.",
                    "existing_file_id": existing_file['file_id'],
                    "existing_status": existing_file['status']
                })
                continue
            
            metadata = {
                "id": file_id,
                "name": original_filename,
                "path": str(file_path.resolve()),
                "size": len(content),
                "type": file.content_type,
                "source_url": "direct_upload"
            }
            
            await db.log_initial_file(metadata, file_hash)

            response_data.append(
                {
                    "id": file_id,
                    "name": original_filename,
                    "path": str(file_path.resolve()),
                    "size": len(content),
                    "type": file.content_type,
                }
            )

        except Exception as e:
            print(f"Error processing file {file.filename}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Could not save file: {file.filename}. Error: {e}",
            )
    print(response_data)
    return {
        "message": f"Processed {len(files)} file(s). New: {len(response_data)}, Duplicates: {len(duplicates_data)}.",
        "new_files": response_data,
        "duplicates": duplicates_data,
    }

async def _process_saved_files(saved_files: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """
    Generator that processes already-saved files and yields SSE events.
    Each element in saved_files = (file_id, file_path).
    """
    for file_info in saved_files:
        file_path = file_info["path"]
        file_id = file_info["id"]
        file_name = file_info["original_name"]

        try:
            processing_result = await process_pdf(file_path)

            if not processing_result or "analysis" not in processing_result:
                error_result = {
                    "fileId": file_id,
                    "fileName": os.path.basename(file_path),
                    "error": "Processing returned no result."
                }
                yield f"data: {json.dumps(error_result)}\n\n"
                continue

            analysis_data = processing_result.get("analysis", {}).get("json", {})
            analysis_result_model = AnalysisResult(**analysis_data)
            if analysis_result_model.classification.lower() == "unstructured":
                analysis_result_model.classification = "Unstructured"
            else:
                analysis_result_model.classification = "Structured"
            print(analysis_result_model.model_dump_json())
            final_result = FileProcessingResult(
                fileId=file_id,
                fileName=os.path.basename(file_path),
                qualityMetrics=QualityMetrics(parseAccuracy=analysis_result_model.quality_score if analysis_result_model.quality_score else 1),
                analysis=analysis_result_model
            )
            await db.log_analysis_result(file_id, final_result.model_dump())

            yield f"data: {final_result.model_dump_json()}\n\n"

        except Exception as e:
            error_detail = f"Processing error: {e}"

            error_result = {
                "fileId": file_id,
                "fileName": os.path.basename(file_path),
                "error": f"Processing error: {e}"
            }
            await db.log_error(file_id, error_detail, status="ANALYSIS_FAILED")

            yield f"data: {json.dumps(error_result)}\n\n"

        finally:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass

@app.post("/process-files")
async def process_files_endpoint(
    files: List[UploadFile] = File(...),
    file_ids: List[str] = Form(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    saved_info = []  # list of dicts with {path, id, original_name}

    for upload, fid in zip(files, file_ids):

        if not upload.filename:
            continue

        suffix = os.path.splitext(upload.filename)[1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=UPLOAD_DIRECTORY)
        tmp_path = tmp.name
        tmp.close()

        try:
            with open(tmp_path, "wb") as buffer:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    buffer.write(chunk)
            await upload.close()

            saved_info.append({
                "path": tmp_path,
                "id": fid,
                "original_name": upload.filename
            })

        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            continue

    return StreamingResponse(
        _process_saved_files(saved_info),
        media_type="text/event-stream"
    )

@app.post("/ingest/", tags=["Ingestion"])
async def start_ingestion_process(
    files: List[UploadFile] = File(...),
    file_details: str = Form(...),
):
    """
    Accepts multiple files and their details, processes them based on their
    classification (Structured or Unstructured), and streams the progress and results back to the client.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided for ingestion.")

    try:
        details_list = json.loads(file_details)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for file_details.")

    async def ingestion_generator():
        total_files = len(details_list)
        for i, details_data in enumerate(details_list):
            filename = details_data.get("name")
            file_id = details_data.get("id")
            full_filepath = details_data.get("path")
            result_payload = {}
            base_name, file_type = os.path.splitext(filename)

            try:
                # **CORE LOGIC: Check classification and route to the correct ingestion path**
                if details_data.get("classification") == "Structured":
                    # --- STRUCTURED FILE PROCESSING ---
                    print(f"Processing STRUCTURED file: {filename}")
                    filename_without_ext, _ = os.path.splitext(str(filename))
                    markdown_file_path = None

                    if 'xl' in file_type or 'csv' in file_type:
                        markdown_file_path, markdown_result = extract_markdown_from_excel_csv(full_filepath)
                    else:
                        markdown_file_path, markdown_result = extract_markdown_from_file(full_filepath)

                    if markdown_file_path is None:
                        raise FileNotFoundError(f"Processed markdown file not found: {markdown_file_path}")

                    if not os.path.exists(markdown_file_path):
                         raise FileNotFoundError(f"Processed markdown file not found: {markdown_file_path}")

                    table_files = extract_markdown_tables(markdown_file_path)
                    table_details_results = []
                    for table_file in table_files:
                        with open(table_file, 'r', encoding='utf-8') as tf:
                            table_markdown = tf.read()
                        
                        # Assuming ingest_markdown_table returns a TableDetails object
                        table_ingest_result = ingest_markdown_table(
                            md_table=table_markdown,
                            file_name=str(filename),
                            file_size=details_data.get('size', 0),
                            intents=details_data.get("analysis", {}).get("intents"),
                            brief_summary=details_data.get("analysis", {}).get("brief_summary"),
                            subdomain=details_data.get("analysis", {}).get("subdomain"),
                            publishing_authority=details_data.get("analysis", {}).get("publishing_authority")
                        )
                        table_details_results.append(table_ingest_result)

                    structured_details = StructuredIngestionDetails(type="structured", tables=table_details_results)
                    ingestion_result_obj = FileIngestionResult(
                        fileName=str(filename),
                        fileId=file_id,
                        fileSize=details_data.get('size', 0),
                        status="success",
                        ingestionDetails=structured_details,
                    )
                    result_payload = ingestion_result_obj.dict()

                else:
                    # --- UNSTRUCTURED FILE PROCESSING ---
                    print(f"Processing UNSTRUCTURED file: {filename}")
                    analysis = details_data.get("analysis", {})
                    ingestion_result_obj = ingest_unstructured_file(
                        file_path=full_filepath,
                        file_name=filename,
                        fileId=file_id,
                        category=analysis.get("subdomain"),
                        reference=analysis.get("publishing_authority"),
                        url=details_data.get("sourceUrl", "default_url"),
                        published_date=analysis.get("published_date")
                    )
                    result_payload = ingestion_result_obj.dict()

                # --- COMMON LOGGING AND PROGRESS UPDATE ---
                # This block runs after either structured or unstructured processing succeeds
                await db.log_ingestion_result(
                    file_id,
                    result_payload.get("ingestionDetails")
                )

            except Exception as e:
                # --- UNIFIED EXCEPTION HANDLING ---
                print(f"\nAn error occurred during ingestion for {filename}: {e}")
                error_detail = f"An unhandled exception occurred: {str(e)}"
                result_payload = FileIngestionResult(
                    fileName=str(filename),
                    fileId=file_id,
                    fileSize=details_data.get('size', 0),
                    status="failed",
                    error=error_detail,
                ).dict()
                if file_id:
                    await db.log_error(file_id, error_detail, status="INGESTION_FAILED")

            finally:
                # --- PROGRESS & CLEANUP ---
                result_payload["progress"] = ((i + 1) / total_files) * 100

                # Cleanup successful files
                if result_payload.get("status") == "success":
                    if full_filepath and os.path.exists(full_filepath):
                        os.remove(full_filepath)
                    filename_without_ext, _ = os.path.splitext(str(filename))
                    md_filepath = os.path.join(MARKDOWN_DIRECTORY, str(filename_without_ext) + '.md')
                    if os.path.exists(md_filepath):
                        os.remove(md_filepath)
                
                # --- YIELD RESULT TO CLIENT ---
                yield f"data: {json.dumps(result_payload)}\n\n"
                await asyncio.sleep(0.01) # Small delay to allow message to be sent

    return StreamingResponse(ingestion_generator(), media_type="text/event-stream")







# @app.post("/ingest/", response_model=IngestionResponse, tags=["Ingestion"])
# async def start_ingestion_process(
#     files: List[UploadFile] = File(...),
#     file_details: str = Form(...),
# ):
#     if not files:
#         raise HTTPException(status_code=400, detail="No files were provided for ingestion.")

#     try:
#         details_list = json.loads(file_details)
#         print(f"Parsed file details: {details_list}")  # Debugging line
#     except json.JSONDecodeError:
#         raise HTTPException(status_code=400, detail="Invalid JSON format for file_details or db_config.")

#     ingestion_results = []
#     files_map = {file.filename: file for file in files}

#     for details_data in details_list:
#         file_ingestion_result = []
#         filename = details_data.get("name")
#         if not filename:
#                 continue
#         filename_without_ext, file_type = os.path.splitext(str(filename))
#         full_filepath = details_data.get("path")
#         analysis = details_data.get("analysis")
#         intents = analysis.get("intents")
#         brief_summary = analysis.get("brief_summary")
#         subdomain = analysis.get("subdomain")
#         publishing_authority = analysis.get("publishing_authority")
#         if details_data.get("classification") == "Structured":
#             try:
#                 file_path = os.path.join(MARKDOWN_DIRECTORY, filename_without_ext+ ".md")
#                 print(f"Processing file: {file_path}")

#                 if not os.path.exists(file_path):
#                     raise FileNotFoundError(f"Processed markdown file not found: {file_path}")

#                 if details_data.get('qualityMetrics').get('complexity')>1:
#                     if 'csv' in file_type or 'xl' in file_type:
#                         with open(file_path, 'r', encoding='utf-8') as tf:
#                             markdown_text = tf.read()
#                         cleaned_md = clean_markdown(markdown_text)
#                         with open(file_path, "w", encoding="utf-8") as f:
#                             f.write(cleaned_md)

#                 table_files = extract_markdown_tables(file_path)
#                 print(f"Extracted {len(table_files)} tables from {file_path}")
#                 table_details = []
#                 for table_file in table_files:
#                     with open(table_file, 'r', encoding='utf-8') as tf:
#                         table_markdown = tf.read()
#                     if details_data.get('qualityMetrics').get('complexity')>1:
#                         table_markdown = csv_to_markdown_file(table_file)
#                     table_ingest_result = ingest_markdown_table(table_markdown, str(filename), details_data.get('size', 0), intents, brief_summary, subdomain, publishing_authority)
                    
#                     print(f"Table ingestion result: {table_ingest_result}")
#                     table_details.append(table_ingest_result)
                    

#                 structured_ingestion_details = StructuredIngestionDetails(type="structured", tables=table_details)
#                 ingestion_results.append(FileIngestionResult(
#                         fileName=str(filename),
#                         fileSize=details_data.get('size', 0),
#                         status="success",
#                         ingestionDetails=structured_ingestion_details,
#                     ))

#             except Exception as e:
#                 ingestion_results.append(
#                     FileIngestionResult(
#                         fileName=str(filename),
#                         fileSize=details_data.get('size', 0),
#                         status="failed",
#                         error=f"An unexpected error occurred: {str(e)}"
#                     )
#                 )
#         else:
#             try:
#                 result_success = ingest_unstructured_file(
#                     file_path=full_filepath,
#                     category=subdomain,
#                     reference=publishing_authority,
#                     url="https://esankhyiki.mospi.gov.in"
#                     )
#                 file_ingestion_result.append(result_success.ingestionDetails)
#                 ingestion_results.append(result_success)
#             except ImportError:
#                 print("\nPlease install fpdf to run the example with a dummy file: pip install fpdf")
#             except Exception as e:
#                 print(f"\nAn error occurred during the example run: {e}")

#             try:
#                 file_path = os.path.join(MARKDOWN_DIRECTORY, filename_without_ext+ ".md")
#                 table_files = extract_markdown_tables(file_path)
#                 print(f"Extracted {len(table_files)} tables from {file_path}")
#                 table_details = []
#                 for table_file in table_files:
#                     with open(table_file, 'r', encoding='utf-8') as tf:
#                         table_markdown = tf.read()
#                     if details_data.get('qualityMetrics').get('complexity')>1:
#                         table_markdown = csv_to_markdown_file(table_file)
#                     table_ingest_result = ingest_markdown_table(table_markdown, str(filename), details_data.get('size', 0), intents, brief_summary, subdomain, publishing_authority)
                    
#                     print(f"Table ingestion result: {table_ingest_result}")
#                     table_details.append(table_ingest_result)
                    

#                 structured_ingestion_details = StructuredIngestionDetails(type="structured", tables=table_details)
#                 file_ingestion_result.append(structured_ingestion_details)
#             except Exception as e:
#                 ingestion_results.append(
#                     FileIngestionResult(
#                         fileName=str(filename),
#                         fileSize=details_data.get('size', 0),
#                         status="failed",
#                         error=f"An unexpected error occurred: {str(e)}"
#                     )
#                 )
#             ingestion_results.append(FileIngestionResult(
#                         fileName=str(filename),
#                         fileSize=details_data.get('size', 0),
#                         status="success",
#                         ingestionDetails=file_ingestion_result,
#                     ))
#     result = IngestionResponse(results=ingestion_results)
#     print(f"Ingestion results: {result}")
#     return result
