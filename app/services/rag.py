# Copyright 2024
# Directory: yt-agentic-rag/app/services/rag.py

"""
RAG (Retrieval-Augmented Generation) service implementation.
Orchestrates the complete RAG pipeline: chunk → embed → search → generate.
"""

import logging
import time
import re
from typing import List, Dict, Any, Tuple
from ..config.database import db
from .embedding import embedding_service
from .chat import chat_service
from .chunker import chunker
from ..data.default_documents import DEFAULT_DOCUMENTS

logger = logging.getLogger(__name__)


class RAGService:
    """Main RAG service orchestrating the complete pipeline."""
    
    def __init__(self):
        """Initialize RAG service."""
        self.db = db
        self.embedding_service = embedding_service
        self.chat_service = chat_service
        self.chunker = chunker
    
    async def seed_documents(self, documents: List[Dict[str, str]] = None) -> int:
        """
        Seed the knowledge base with documents.
        
        Args:
            documents: Optional list of documents. If None, uses default documents.
            
        Returns:
            Number of chunks successfully inserted
        """
        start_time = time.time()
        
        # Use default documents if none provided
        if documents is None:
            documents = DEFAULT_DOCUMENTS
            logger.info("Using default documents for seeding")
        
        try:
            # Step 1: Chunk documents
            chunks = self.chunker.chunk_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
            
            # Step 2: Generate embeddings for all chunks
            texts = [chunk['text'] for chunk in chunks]
            embeddings = await self.embedding_service.embed_texts(texts)
            
            # Step 3: Combine chunks with embeddings
            for chunk, embedding in zip(chunks, embeddings):
                chunk['embedding'] = embedding
            
            # Step 4: Store in database
            inserted_count = await self.db.upsert_chunks(chunks)
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Seeding completed in {elapsed_ms}ms: {inserted_count} chunks inserted")
            
            return inserted_count
            
        except Exception as e:
            logger.error(f"Seeding failed: {e}")
            raise
    
    async def answer_query(self, query: str, top_k: int = 6) -> Dict[str, Any]:
        """
        Process a query through the complete RAG pipeline.
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer, citations, and debug info
        """
        start_time = time.time()
        
        try:
            # Step 1: Generate query embedding
            query_embedding = await self.embedding_service.embed_query(query)
            
            # Step 2: Vector similarity search
            search_results = await self.db.vector_search(query_embedding, top_k)
            
            if not search_results:
                return {
                    'text': "I don't have enough information to answer your question. Could you please rephrase or provide more context?",
                    'citations': [],
                    'debug': {
                        'top_doc_ids': [],
                        'latency_ms': int((time.time() - start_time) * 1000)
                    }
                }
            
            # Step 3: Deduplicate and prepare context
            context_blocks = self._prepare_context(search_results)
            
            # Step 4: Generate answer
            answer_text = await self.chat_service.generate_answer(query, context_blocks)
            
            # Step 5: Extract citations from answer
            citations = self._extract_citations(answer_text, context_blocks)
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            result = {
                'text': answer_text,
                'citations': citations,
                'debug': {
                    'top_doc_ids': [block['chunk_id'] for block in context_blocks],
                    'latency_ms': elapsed_ms
                }
            }
            
            logger.info(f"Query processed in {elapsed_ms}ms with {len(citations)} citations")
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                'text': f"I encountered an error while processing your question: {str(e)}",
                'citations': [],
                'debug': {
                    'top_doc_ids': [],
                    'latency_ms': int((time.time() - start_time) * 1000)
                }
            }
    
    def _prepare_context(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare context blocks from search results with simple deduplication.
        
        Args:
            search_results: Raw search results from database
            
        Returns:
            Processed context blocks
        """
        # Simple deduplication by chunk_id prefix (MMR-lite)
        seen_prefixes = set()
        context_blocks = []
        
        for result in search_results:
            chunk_id = result.get('chunk_id', '')
            
            # Extract base chunk ID (before #)
            base_id = chunk_id.split('#')[0] if '#' in chunk_id else chunk_id
            
            if base_id not in seen_prefixes:
                context_blocks.append(result)
                seen_prefixes.add(base_id)
            
            # Limit to reasonable context size
            if len(context_blocks) >= 4:
                break
        
        logger.info(f"Prepared {len(context_blocks)} unique context blocks")
        return context_blocks
    
    def _extract_citations(self, answer_text: str, context_blocks: List[Dict[str, Any]]) -> List[str]:
        """
        Extract citation chunk_ids from the generated answer.
        
        Args:
            answer_text: Generated answer text
            context_blocks: Context blocks that were provided
            
        Returns:
            List of chunk_ids that were cited
        """
        # Find all citations in format [chunk_id]
        citation_pattern = r'\[([^\]]+)\]'
        found_citations = re.findall(citation_pattern, answer_text)
        
        # Filter to only include valid chunk_ids from context
        valid_chunk_ids = {block['chunk_id'] for block in context_blocks}
        valid_citations = [cite for cite in found_citations if cite in valid_chunk_ids]
        
        # Remove duplicates while preserving order
        unique_citations = []
        for cite in valid_citations:
            if cite not in unique_citations:
                unique_citations.append(cite)
        
        return unique_citations


# Global RAG service instance
rag_service = RAGService()
