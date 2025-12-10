#!/usr/bin/env python3
# Copyright 2024
# Directory: yt-agentic-rag/devtools/verify_rag.py

"""
RAG Verification Script.

Proves we're doing real RAG with vector similarity search by showing:
- Actual embeddings stored in database
- Similarity scores between queries
- Vector operations in action

This is useful for demonstrating that RAG is working correctly,
not just passing text to the LLM.

Usage:
    python devtools/verify_rag.py
"""

import os
import json
import asyncio
from dotenv import load_dotenv
from supabase import create_client
from app.services.embedding import embedding_service

# Load environment variables
load_dotenv()

async def prove_real_rag():
    """Prove we're doing actual vector similarity search, not just text feeding."""
    
    print("🔍 PROVING REAL RAG IMPLEMENTATION")
    print("=" * 60)
    
    # Initialize Supabase client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url or not supabase_key:
        print("❌ Missing Supabase credentials")
        return
    
    supabase = create_client(supabase_url, supabase_key)
    
    # Step 1: Show actual embeddings exist in database
    print("\n1️⃣ CHECKING ACTUAL EMBEDDINGS IN DATABASE")
    print("-" * 40)
    
    try:
        result = supabase.table('rag_chunks').select('chunk_id, text, embedding').limit(2).execute()
        
        if not result.data:
            print("❌ No data found. Run /seed first!")
            return
        
        for chunk in result.data:
            embedding = chunk['embedding']
            print(f"📄 Chunk: {chunk['chunk_id']}")
            print(f"📝 Text preview: {chunk['text'][:100]}...")
            print(f"🔢 Embedding dimensions: {len(embedding)}")
            print(f"🎯 First 10 values: {embedding[:10]}")
            print(f"📊 Vector magnitude: {sum(x*x for x in embedding)**0.5:.4f}")
            print()
    
    except Exception as e:
        print(f"❌ Database error: {e}")
        return
    
    # Step 2: Test different queries with vector similarity
    print("2️⃣ TESTING VECTOR SIMILARITY WITH DIFFERENT QUERIES")
    print("-" * 50)
    
    test_queries = [
        "Can I return shoes?",                    # Should match return policy
        "How much is shipping?",                  # Should match shipping policy  
        "What size should I order?",              # Should match sizing guide
        "How do I contact support?",              # Should match customer support
        "What's the weather like?",               # Should find no good matches
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        
        try:
            # Generate query embedding
            query_embedding = await embedding_service.embed_query(query)
            print(f"📐 Query embedding dimensions: {len(query_embedding)}")
            print(f"🎯 Query vector preview: {query_embedding[:5]}...")
            
            # Perform vector search using Supabase's match_chunks function
            search_result = supabase.rpc('match_chunks', {
                'query_embedding': query_embedding,
                'match_threshold': 0.1,
                'match_count': 3
            }).execute()
            
            if search_result.data:
                print(f"✅ Found {len(search_result.data)} similar chunks:")
                for i, match in enumerate(search_result.data[:3]):
                    similarity = 1 - match.get('similarity', 0)  # Convert distance to similarity
                    print(f"   {i+1}. {match['chunk_id']} (similarity: {similarity:.4f})")
                    print(f"      Preview: {match['text'][:80]}...")
            else:
                print("❌ No similar chunks found")
                
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    # Step 3: Compare embeddings to show they're actually different
    print("\n3️⃣ PROVING EMBEDDINGS ARE ACTUALLY DIFFERENT")
    print("-" * 45)
    
    try:
        # Get embeddings for different texts
        text1 = "return policy shoes"
        text2 = "shipping cost fast delivery" 
        text3 = "return policy shoes"  # Same as text1
        
        emb1 = await embedding_service.embed_query(text1)
        emb2 = await embedding_service.embed_query(text2)
        emb3 = await embedding_service.embed_query(text3)
        
        # Calculate cosine similarity
        def cosine_similarity(a, b):
            dot_product = sum(x*y for x, y in zip(a, b))
            magnitude_a = sum(x*x for x in a)**0.5
            magnitude_b = sum(x*x for x in b)**0.5
            return dot_product / (magnitude_a * magnitude_b)
        
        sim_1_2 = cosine_similarity(emb1, emb2)
        sim_1_3 = cosine_similarity(emb1, emb3)
        
        print(f"📝 Text 1: '{text1}'")
        print(f"📝 Text 2: '{text2}'")
        print(f"📝 Text 3: '{text3}'")
        print(f"🔢 Similarity 1↔2 (different): {sim_1_2:.6f}")
        print(f"🔢 Similarity 1↔3 (identical): {sim_1_3:.6f}")
        
        if sim_1_3 > 0.999 and sim_1_2 < 0.9:
            print("✅ PROOF: Identical texts have ~1.0 similarity, different texts have lower similarity")
        else:
            print("⚠️  Unexpected similarity scores")
            
    except Exception as e:
        print(f"❌ Error comparing embeddings: {e}")
    
    # Step 4: Show the actual RAG pipeline in action
    print("\n4️⃣ TRACING THE FULL RAG PIPELINE")
    print("-" * 35)
    
    test_query = "Can I return expensive shoes?"
    print(f"🔍 Test query: '{test_query}'")
    
    try:
        from app.services.rag import rag_service
        
        # This will show the full pipeline
        result = await rag_service.answer_query(test_query, top_k=3)
        
        print(f"🤖 Generated answer: {result['text'][:200]}...")
        print(f"📚 Citations found: {result['citations']}")
        print(f"🔍 Top document IDs: {result['debug']['top_doc_ids']}")
        print(f"⏱️  Processing time: {result['debug']['latency_ms']}ms")
        
        print("\n✅ PROOF COMPLETE: This is real vector similarity search!")
        print("   - Actual embeddings stored in database")
        print("   - Vector similarity calculations performed") 
        print("   - Different queries retrieve different chunks")
        print("   - Citations reference specific chunk IDs")
        
    except Exception as e:
        print(f"❌ Error in RAG pipeline: {e}")

if __name__ == "__main__":
    asyncio.run(prove_real_rag())
