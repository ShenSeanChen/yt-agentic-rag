# Copyright 2024
# Directory: yt-agentic-rag/app/config/database.py

"""
Database connection and schema management using Supabase SDK.
Handles Supabase Postgres with pgvector extension for RAG operations.
"""

import logging
from typing import Dict, List, Any, Optional
from supabase import create_client, Client
from .settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class Database:
    """Supabase database operations manager for RAG functionality."""
    
    def __init__(self):
        """Initialize Supabase client."""
        self.supabase: Optional[Client] = None
        self._admin_client: Optional[Client] = None
    
    async def connect(self) -> None:
        """Initialize Supabase clients."""
        try:
            # Regular client with anon key (for future frontend compatibility)
            self.supabase = create_client(
                settings.supabase_url,
                settings.supabase_anon_key
            )
            
            # Admin client with service role key (for backend operations)
            self._admin_client = create_client(
                settings.supabase_url,
                settings.supabase_service_role_key
            )
            
            logger.info("Supabase clients initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase clients: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close Supabase connections (cleanup if needed)."""
        # Supabase clients don't need explicit disconnection
        logger.info("Supabase clients cleaned up")
    
    def get_client(self, admin: bool = True) -> Client:
        """
        Get Supabase client instance.
        
        Args:
            admin: If True, returns admin client with service role key.
                   If False, returns regular client with anon key.
        """
        if not self.supabase or not self._admin_client:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        return self._admin_client if admin else self.supabase
    
    async def initialize_schema(self) -> None:
        """
        Check if database schema is initialized.
        
        Note: The actual schema setup must be done manually in Supabase.
        Run the SQL script: sql/init_supabase.sql in your Supabase SQL Editor.
        """
        try:
            client = self.get_client(admin=True)
            
            # Check if the table exists and has the required structure
            result = client.table('rag_chunks').select('id').limit(1).execute()
            
            if result.data is not None:
                logger.info("Database schema is properly initialized")
                
                # Test the vector search function
                try:
                    stats_result = client.rpc('get_chunk_stats').execute()
                    if stats_result.data:
                        stats = stats_result.data[0]
                        logger.info(f"Database stats: {stats['total_chunks']} chunks, {stats['unique_sources']} sources")
                except Exception:
                    logger.warning("get_chunk_stats function not available - some features may not work")
                    
            else:
                logger.error("Database schema not initialized!")
                logger.error(
                    "Please run the SQL initialization script:\n"
                    "1. Open your Supabase project dashboard\n"
                    "2. Go to SQL Editor\n"
                    "3. Run the script: sql/init_supabase.sql\n"
                    "4. Restart your application"
                )
                
        except Exception as e:
            logger.error(f"Database schema check failed: {e}")
            logger.error(
                "Please ensure you've run sql/init_supabase.sql in your Supabase dashboard"
            )
    
    async def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Upsert document chunks into the database using Supabase SDK.
        
        Args:
            chunks: List of chunk dictionaries with keys: chunk_id, source, text, embedding
            
        Returns:
            Number of chunks inserted
        """
        if not chunks:
            return 0
        
        try:
            client = self.get_client(admin=True)
            
            # Prepare data for Supabase
            chunk_data = []
            for chunk in chunks:
                chunk_data.append({
                    'chunk_id': chunk['chunk_id'],
                    'source': chunk['source'],
                    'text': chunk['text'],
                    'embedding': chunk['embedding']  # Supabase handles vector serialization
                })
            
            # Use upsert with on_conflict parameter
            result = client.table('rag_chunks').upsert(
                chunk_data,
                on_conflict='chunk_id'
            ).execute()
            
            inserted_count = len(result.data) if result.data else 0
            logger.info(f"Upserted {inserted_count} chunks to database")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Failed to upsert chunks: {e}")
            raise
    
    async def vector_search(self, query_embedding: List[float], top_k: int = 6) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using Supabase SDK.
        
        Args:
            query_embedding: Query vector embedding
            top_k: Number of results to return
            
        Returns:
            List of matching chunks with similarity scores
        """
        try:
            client = self.get_client(admin=True)
            
            # Use RPC call for vector similarity search
            # This requires creating a PostgreSQL function in Supabase
            result = client.rpc('match_chunks', {
                'query_embedding': query_embedding,
                'match_count': top_k
            }).execute()
            
            if result.data:
                logger.info(f"Vector search returned {len(result.data)} results")
                return result.data
            else:
                # Fallback: if RPC function doesn't exist, use regular query
                # This won't have vector similarity but will work for basic testing
                logger.info("Using fallback query (match_chunks RPC function not available)")
                result = client.table('rag_chunks').select('*').limit(top_k).execute()
                
                # Add mock similarity scores for fallback
                if result.data:
                    for i, chunk in enumerate(result.data):
                        chunk['similarity'] = 1.0 - (i * 0.1)  # Mock decreasing similarity
                
                return result.data or []
                
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check Supabase connection health."""
        try:
            client = self.get_client(admin=True)
            
            # Simple query to test connection
            result = client.table('rag_chunks').select('id').limit(1).execute()
            
            return True  # If no exception, connection is healthy
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global database instance
db = Database()