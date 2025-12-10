# 🛠️ Development Tools

This folder contains helper utilities for development, testing, and debugging.
These are **not part of the production application** - they're tools to help you build and test.

## Tools Overview

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `test_setup.py` | Verify complete setup | After initial setup, before starting server |
| `chatbot.py` | Terminal chat client | Testing the API without a browser |
| `verify_rag.py` | Prove RAG is working | Demonstrating vector search functionality |
| `check_dimensions.py` | Debug embeddings | Troubleshooting vector dimension issues |

---

## 📋 test_setup.py

**Purpose**: Validates your entire setup without starting the server.

```bash
python devtools/test_setup.py
```

**What it tests**:
1. ✅ Module imports
2. ✅ Environment variables
3. ✅ Database connection
4. ✅ Schema validation
5. ✅ Document seeding
6. ✅ RAG query functionality

**When to use**: Run this after completing setup, before your first `uvicorn` start.

---

## 💬 chatbot.py

**Purpose**: Beautiful terminal-based chat interface for the API.

```bash
# Default (localhost:8000)
python devtools/chatbot.py

# Custom URL
python devtools/chatbot.py http://your-server:8000
```

**Features**:
- Rich terminal UI with colors
- Command support (`/help`, `/seed`, `/health`, `/stats`, `/quit`)
- Response time tracking
- Citation display

**When to use**: Quick testing without opening a browser.

---

## 🔍 verify_rag.py

**Purpose**: Demonstrates that real vector similarity search is happening.

```bash
python devtools/verify_rag.py
```

**What it shows**:
1. Actual embeddings stored in database
2. Embedding dimensions and values
3. Similarity scores for different queries
4. Proof that identical texts have similarity ~1.0

**When to use**: 
- Demonstrating RAG to others
- Verifying vector search is working
- Educational purposes

---

## 📐 check_dimensions.py

**Purpose**: Debug tool for checking embedding dimensions in the database.

```bash
python devtools/check_dimensions.py
```

**What it checks**:
- Number of chunks in database
- Embedding vector dimensions
- First few embedding values

**When to use**: 
- Debugging "dimension mismatch" errors
- Verifying embeddings were stored correctly
- After changing embedding models

---

## 💡 Tips

1. **Always activate your virtual environment first**:
   ```bash
   source venv/bin/activate
   ```

2. **Make sure `.env` is configured** before running any script

3. **Start the server first** for `chatbot.py`:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

4. **Run from project root**, not from inside `devtools/`:
   ```bash
   # ✅ Correct
   python devtools/test_setup.py
   
   # ❌ Wrong
   cd devtools && python test_setup.py
   ```

