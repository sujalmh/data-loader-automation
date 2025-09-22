from pydantic import BaseModel, Field, HttpUrl
from typing import List, Union, Literal, Optional, Dict, Any

class UrlListRequest(BaseModel):
    urls: List[HttpUrl]

class QualityMetrics(BaseModel):
    """Defines the structure for data quality metrics."""
    parseAccuracy: float

class AnalysisResult(BaseModel):
    """
    Defines the structure for the analysis object returned by the LLM.
    This model is flexible to handle potential variations in LLM output.
    """
    file_name: Optional[str] = None
    domain: Optional[str] = None
    subdomain: Optional[str] = None
    intents: Optional[Union[List[str], str]] = None
    publishing_authority: Optional[str] = None
    published_date: Optional[str] = None
    period_of_reference: Optional[str] = None
    brief_summary: Optional[str] = None
    document_size: Optional[str] = None
    extra_fields: Dict[str, Any] = {}
    classification: str = "Unstructured"
    quality_score: Optional[Union[float, int]] = None
    error: Optional[str] = None

class FileProcessingResult(BaseModel):
    """
    Defines the final response structure for each processed file.
    """
    fileId: str
    fileName: str
    qualityMetrics: QualityMetrics
    analysis: AnalysisResult

class ColumnSchema(BaseModel):
    """Defines the schema for a single column in a structured table."""
    name: str
    type: str
    primary: Optional[bool] = False

class TableDetails(BaseModel):
    """Holds all the details for a single table extracted from a structured file."""
    tableName: str
    schema_details: List[ColumnSchema]
    rowsInserted: int
    sqlCommands: List[str]
    fileSelectorPrompt: Optional[str] = None

class StructuredIngestionDetails(BaseModel):
    """Details for a successfully ingested structured file, supporting multiple tables."""
    type: Literal["structured"]
    tables: List[TableDetails]
 
class UnstructuredIngestionDetails(BaseModel):
    """Details for a successfully ingested unstructured file."""
    type: Literal["unstructured"]
    collection: str
    chunksCreated: int
    embeddingsGenerated: int
    chunkingMethod: str
    embeddingModel: str

# A union of all possible ingestion detail types
IngestionDetails = Union[StructuredIngestionDetails, UnstructuredIngestionDetails]

class FileIngestionResult(BaseModel):
    """Represents the result of processing a single file."""
    fileName: str
    fileId: str
    fileSize: int
    status: Literal["success", "failed"]
    ingestionDetails: Optional[Union[IngestionDetails, List[IngestionDetails]]] = None
    error: Optional[str] = None

class IngestionResponse(BaseModel):
    """The final response object returned by the API."""
    results: List[FileIngestionResult]

class FileMetadata(BaseModel):
    """Defines the structure of the metadata for a successfully fetched file."""
    name: str
    path: str
    size: int
    type: str
    source_url: str
    file_base64: Optional[str] = None

class DownloadSuccess(BaseModel):
    """Wrapper for a successful download result."""
    status: str = "success"
    id: str
    url: HttpUrl
    data: FileMetadata

class DownloadDuplicate(BaseModel):
    """Wrapper for a duplicate file result."""
    status: str = "duplicate"
    url: HttpUrl
    message: str
    existing_file_id: str
    existing_status: str

class DownloadError(BaseModel):
    """Wrapper for a failed download result."""
    status: str = "error"
    url: HttpUrl
    error: str

DownloadResult = Union[DownloadSuccess, DownloadError, DownloadDuplicate]
