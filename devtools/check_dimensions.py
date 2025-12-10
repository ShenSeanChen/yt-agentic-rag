#!/usr/bin/env python3
# Copyright 2024
# Directory: yt-agentic-rag/devtools/check_dimensions.py

"""
Debug Utility: Check Embedding Dimensions.

Verifies that embeddings in the Supabase database have the correct dimensions.
Useful for debugging vector search issues.

Usage:
    python devtools/check_dimensions.py
"""

import os
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables
load_dotenv()

def check_embedding_dimensions():
    """Check the dimensions of embeddings in the database."""
    
    # Initialize Supabase client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url or not supabase_key:
        print("❌ Missing Supabase credentials in .env file")
        return
    
    supabase = create_client(supabase_url, supabase_key)
    
    try:
        # Method 1: Check using raw SQL
        print("🔍 Checking embedding dimensions...")
        
        # Get one row to check dimensions
        result = supabase.rpc('match_chunks', {
            'query_embedding': '[0]' * 1536,  # Dummy vector
            'match_threshold': 0.1,
            'match_count': 1
        }).execute()
        
        if result.data:
            print(f"✅ Found {len(result.data)} chunks in database")
        else:
            print("⚠️  No chunks found in database. Run /seed first.")
            return
        
        # Method 2: Direct table query to check schema
        schema_result = supabase.table('rag_chunks').select('*').limit(1).execute()
        
        if schema_result.data:
            chunk = schema_result.data[0]
            if 'embedding' in chunk and chunk['embedding']:
                dimensions = len(chunk['embedding'])
                print(f"📐 Embedding dimensions: {dimensions}")
                print(f"📊 Vector type: {type(chunk['embedding'])}")
                print(f"🎯 First few values: {chunk['embedding'][:5]}...")
            else:
                print("❌ No embedding data found")
        
        # Method 3: Check table schema info
        print("\n📋 Table Schema Info:")
        print("Run this SQL in Supabase SQL Editor:")
        print("SELECT column_name, data_type FROM information_schema.columns")
        print("WHERE table_name = 'rag_chunks' AND column_name = 'embedding';")
        
        print("\n🔧 Or use pgvector function:")
        print("SELECT vector_dims(embedding) FROM rag_chunks LIMIT 1;")
        
    except Exception as e:
        print(f"❌ Error checking dimensions: {e}")
        print("💡 Make sure you've run the /seed endpoint first to populate data")

if __name__ == "__main__":
    check_embedding_dimensions()
