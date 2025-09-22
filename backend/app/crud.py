import asyncpg
import json
from uuid import UUID

from app.models.model_definition import QualityMetrics, AnalysisResult

async def create_file_record(pool: asyncpg.Pool, filename: str, path: str, size: int) -> UUID:
    """Inserts a new file record into the 'files' table and returns its UUID."""
    query = """
        INSERT INTO files (original_filename, stored_path, file_size_bytes)
        VALUES ($1, $2, $3)
        RETURNING file_id;
    """
    file_id = await pool.fetchval(query, filename, path, size)
    return file_id

async def get_file_by_filename(pool: asyncpg.Pool, filename: str):
    """Retrieves a file record by its original filename."""
    query = "SELECT * FROM files WHERE original_filename = $1 LIMIT 1;"
    record = await pool.fetchrow(query, filename)
    return record

async def add_file_analysis(
    pool: asyncpg.Pool,
    file_id: UUID,
    classification: str,
    quality_metrics: QualityMetrics,
    analysis: AnalysisResult
):
    """Updates the file record with metrics and adds a new record to file_analysis."""
    async with pool.acquire() as connection:
        async with connection.transaction():
            # 1. Update the 'files' table
            update_files_query = """
                UPDATE files
                SET classification = $1, parse_accuracy = $2, complexity_score = $3
                WHERE file_id = $4;
            """
            await connection.execute(
                update_files_query,
                classification,
                quality_metrics.parseAccuracy,
                quality_metrics.complexity,
                file_id
            )

            # 2. Insert into the 'file_analysis' table
            insert_analysis_query = """
                INSERT INTO file_analysis (
                    file_id, title, brief_summary, subdomain, publishing_authority,
                    language, geographical_coverage, temporal_coverage, intents
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12);
            """
            await connection.execute(
                insert_analysis_query,
                file_id,
                analysis.title,
                analysis.brief_summary,
                analysis.subdomain,
                analysis.publishing_authority,
                analysis.language,
                json.dumps(analysis.geographical_coverage),
                json.dumps(analysis.temporal_coverage),
                json.dumps(analysis.intents),
            )

async def create_ingestion_log(pool: asyncpg.Pool, file_id: UUID, ingestion_type: str, db_config: dict) -> UUID:
    """Creates a new ingestion log entry and returns its UUID."""
    query = """
        INSERT INTO ingestion_log (file_id, status, ingestion_type, db_config_details)
        VALUES ($1, 'in_progress', $2, $3)
        RETURNING ingestion_id;
    """
    ingestion_id = await pool.fetchval(query, file_id, ingestion_type, json.dumps(db_config))
    return ingestion_id

async def update_ingestion_log_status(pool: asyncpg.Pool, ingestion_id: UUID, status: str, error: str = None):
    """Updates the status and optional error message of an ingestion log entry."""
    query = """
        UPDATE ingestion_log
        SET status = $1, error_message = $2
        WHERE ingestion_id = $3;
    """
    await pool.execute(query, status, error, ingestion_id)

async def create_ingested_object(pool: asyncpg.Pool, ingestion_id: UUID, obj_type: str, obj_name: str, metadata: dict):
    """Logs a successfully created object (e.g., a database table) after ingestion."""
    query = """
        INSERT INTO ingested_objects (ingestion_id, object_type, object_name, metadata)
        VALUES ($1, $2, $3, $4);
    """
    await pool.execute(query, ingestion_id, obj_type, obj_name, json.dumps(metadata))