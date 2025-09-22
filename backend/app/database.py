import os
import asyncpg
from dotenv import load_dotenv
from fastapi import HTTPException

# Load environment variables from .env file
load_dotenv()

class Database:
    pool: asyncpg.Pool = None

async def connect_to_db():
    """Creates a connection pool during application startup."""
    try:
        Database.pool = await asyncpg.create_pool(
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            min_size=1,
            max_size=10
        )
        print("Successfully connected to PostgreSQL and created connection pool.")
    except Exception as e:
        print(f"FATAL: Could not connect to PostgreSQL database: {e}")
        # In a real app, you might want to prevent startup if the DB is unavailable.

async def close_db_connection():
    """Closes the connection pool during application shutdown."""
    if Database.pool:
        await Database.pool.close()
        print("PostgreSQL connection pool closed.")

async def get_db_pool():
    """Dependency to get the database pool in endpoint functions."""
    if Database.pool is None:
        raise HTTPException(status_code=503, detail="Database connection pool is not available.")
    return Database.pool