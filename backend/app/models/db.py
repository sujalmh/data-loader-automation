import os
import asyncpg
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global connection pool
_pool = None


async def get_db_pool():
    """
    Returns the existing database connection pool.
    Raises an exception if the pool is not initialized.
    """
    global _pool
    if _pool is None:
        raise Exception("Database connection pool is not initialized. Call connect_db() first.")
    return _pool

async def connect_db():
    global _pool
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable not set.")

    try:
        _pool = await asyncpg.create_pool(dsn=db_url)
        print("✅ Database connection pool created successfully.")
    except asyncpg.InvalidCatalogNameError:
        print("⚠️ Database not found. Creating it...")

        # Parse URL and connect to 'postgres' default DB
        from urllib.parse import urlparse
        parsed = urlparse(db_url)

        # Rebuild a connection string pointing to 'postgres'
        db_user = parsed.username
        db_pass = parsed.password or ""
        db_host = parsed.hostname
        db_port = parsed.port or 5432
        db_name = parsed.path.lstrip("/")

        admin_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/postgres"
        conn = await asyncpg.connect(admin_url)
        await conn.execute(f'CREATE DATABASE "{db_name}" OWNER "{db_user}";')
        await conn.close()

        # Retry creating pool
        _pool = await asyncpg.create_pool(dsn=db_url)
        print("✅ Database created and connection pool initialized.")


async def disconnect_db():
    """
    Closes the database connection pool. This should be called on application shutdown.
    """
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        print("✅ Database connection pool closed.")


async def init_db():
    """
    Initializes the database by creating the necessary tables and functions.
    """
    db_pool = await get_db_pool()
    async with db_pool.acquire() as connection:
        # Create a function to automatically update the 'updated_at' timestamp
        await connection.execute("""
            CREATE OR REPLACE FUNCTION trigger_set_timestamp()
            RETURNS TRIGGER AS $$
            BEGIN
              NEW.updated_at = NOW();
              RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """)

        # Create the main table for storing file processing records
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS  file_records (
                file_id TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size BIGINT,
                file_type TEXT,
                source_url TEXT,
                status TEXT NOT NULL,
                analysis_result JSONB,
                ingestion_details JSONB,
                error_message TEXT,
                created_at TIMESTAMPTZ DEFAULT now(),
                updated_at TIMESTAMPTZ DEFAULT now()
            );
        """)

        # Drop the trigger if it exists
        await connection.execute("DROP TRIGGER IF EXISTS set_timestamp ON file_records;")

        # Create the trigger to update 'updated_at' on any row update
        await connection.execute("""
            CREATE TRIGGER set_timestamp
            BEFORE UPDATE ON file_records
            FOR EACH ROW
            EXECUTE PROCEDURE trigger_set_timestamp();
        """)

        print("✅ Database initialized successfully.")

# --- CRUD Functions ---

async def log_initial_file(metadata, file_hash, status="DOWNLOADED"):
    """
    Creates a new record for a downloaded or uploaded file.
    """
    pool = await get_db_pool()
    async with pool.acquire() as connection:
        await connection.execute(
            """
            INSERT INTO file_records (
                file_id, file_hash, file_name, file_path, file_size, file_type, source_url, status
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (file_id) DO NOTHING;
            """,
            metadata['id'],
            file_hash,
            metadata['name'],
            metadata['path'],
            metadata['size'],
            metadata['type'],
            metadata.get('source_url', ''),
            status
        )

async def log_analysis_result(file_id, analysis_data, status="ANALYZED"):
    """
    Updates a file record with the analysis results.
    """
    pool = await get_db_pool()
    analysis_json = json.dumps(analysis_data)
    async with pool.acquire() as connection:
        await connection.execute(
            """
            UPDATE file_records
            SET analysis_result = $1, status = $2
            WHERE file_id = $3;
            """,
            analysis_json,
            status,
            file_id
        )

async def log_ingestion_result(file_id, ingestion_data, status="INGESTED"):
    """
    Updates a file record with the final ingestion results.
    """
    pool = await get_db_pool()
    ingestion_json = json.dumps(ingestion_data)
    async with pool.acquire() as connection:
        await connection.execute(
            """
            UPDATE file_records
            SET ingestion_details = $1, status = $2
            WHERE file_id = $3;
            """,
            ingestion_json,
            status,
            file_id
        )

async def log_error(file_id, error_message, status="FAILED"):
    """
    Updates a file record to log an error.
    """
    pool = await get_db_pool()
    async with pool.acquire() as connection:
        await connection.execute(
            """
            UPDATE file_records
            SET error_message = $1, status = $2
            WHERE file_id = $3;
            """,
            error_message,
            status,
            file_id
        )

async def check_duplicate(file_hash: str):
    """
    Checks if a file with the given hash already exists in the database.
    Returns the existing file_id and status if found, otherwise None.
    """
    pool = await get_db_pool()
    async with pool.acquire() as connection:
        # Fetch the record if the hash matches
        record = await connection.fetchrow(
            "SELECT file_id, status FROM file_records WHERE file_hash = $1",
            file_hash
        )
        # Return the record as a dictionary if found, otherwise None
        return dict(record) if record else None