# Directory: yt-agentic-rag/app/agents/orchestrator.py

"""
Agent Orchestrator - The Core of Agentic RAG.

This module implements the main agent reasoning loop:
    Query → RAG Retrieve → Reason → Decide → Act (Tools) → Respond

The orchestrator combines:
- RAG context retrieval from Supabase vector database
- LLM reasoning with tool-calling capabilities (OpenAI/Anthropic)
- Tool execution via the tool registry
- Final response generation with citations

This is what makes "Agentic RAG" different from plain RAG - the agent
can reason about whether to use tools and execute real-world actions.
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional
import openai
import anthropic

from ..config.settings import get_settings
from ..config.database import db
from ..services.embedding import embedding_service
from .tools import tool_registry
from ..schemas.tool_schemas import TOOL_DEFINITIONS

logger = logging.getLogger(__name__)
settings = get_settings()


class AgentService:
    """
    Main agent service that orchestrates the Agentic RAG pipeline.
    
    The agent follows this pipeline:
    1. Retrieve relevant context from RAG (Supabase vector search)
    2. Build messages with system prompt including RAG context
    3. Call LLM with tool definitions
    4. If LLM requests tool calls, execute them and loop back
    5. Generate final response with citations and tool results
    
    This allows the agent to both answer questions using RAG context
    AND take actions (schedule meetings, send emails) when needed.
    """
    
    def __init__(self):
        """Initialize the agent service with configured AI provider."""
        self.provider = settings.ai_provider
        
        if self.provider == "openai":
            self.client = openai.OpenAI(api_key=settings.openai_api_key)
            self.model = settings.openai_chat_model
        elif self.provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            self.model = settings.anthropic_chat_model
        else:
            raise ValueError(f"Unsupported AI provider: {self.provider}")
        
        logger.info(
            f"Agent service initialized with {self.provider} ({self.model})"
        )
    
    async def process_query(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        user_id: Optional[str] = None,
        top_k: int = 6,
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Process a user query through the full agentic pipeline.
        
        Args:
            query: User's question or action request
            chat_history: Optional list of previous messages for context
            user_id: Optional user identifier for personalization
            top_k: Number of RAG chunks to retrieve
            max_iterations: Maximum tool call iterations to prevent infinite loops
            
        Returns:
            Dict containing:
            - text: Final response text
            - tool_calls: List of tool calls made
            - tool_results: Results from tool executions
            - citations: RAG chunk IDs cited in response
            - debug: Debug information
        """
        start_time = time.time()
        tool_calls_made: List[Dict[str, Any]] = []
        tool_results: List[Dict[str, Any]] = []
        
        try:
            # Step 1: Retrieve RAG context
            logger.info(f"Step 1: Retrieving RAG context for: '{query[:50]}...'")
            rag_context = await self._get_rag_context(query, top_k)
            
            # Step 2: Build initial messages with system prompt and chat history
            messages = self._build_initial_messages(query, rag_context, chat_history)
            
            # Step 3: Agent reasoning loop
            iteration = 0
            final_response = None
            
            while iteration < max_iterations:
                iteration += 1
                logger.info(f"Agent iteration {iteration}/{max_iterations}")
                
                # Call LLM with tools
                response = await self._call_llm_with_tools(messages)
                
                # Check if model wants to call tools
                if response.get('tool_calls'):
                    # Execute each tool call
                    for tool_call in response['tool_calls']:
                        tool_name = tool_call['function']['name']
                        tool_args = json.loads(tool_call['function']['arguments'])
                        
                        logger.info(f"Executing tool: {tool_name}")
                        tool_calls_made.append({
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "call_id": tool_call['id']
                        })
                        
                        # Execute the tool via registry
                        result = await tool_registry.execute_tool(
                            tool_name, 
                            **tool_args
                        )
                        tool_results.append({
                            "call_id": tool_call['id'],
                            "tool_name": tool_name,
                            **result
                        })
                        
                        # Add assistant message with tool call
                        # Include 'type' field required by OpenAI API
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": tool_call['id'],
                                "type": "function",
                                "function": tool_call['function']
                            }]
                        })
                        
                        # Add tool result message
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call['id'],
                            "content": json.dumps(result)
                        })
                else:
                    # No more tool calls - we have the final response
                    final_response = response.get('content', '')
                    break
            
            # Step 4: Extract citations from response
            citations = self._extract_citations(final_response or '', rag_context)
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            return {
                "text": final_response or "I processed your request.",
                "tool_calls": tool_calls_made,
                "tool_results": tool_results,
                "citations": citations,
                "debug": {
                    "rag_context_used": len(rag_context) > 0,
                    "rag_chunk_ids": [c['chunk_id'] for c in rag_context],
                    "tools_called": [t['tool_name'] for t in tool_calls_made],
                    "iterations": iteration,
                    "latency_ms": elapsed_ms
                }
            }
            
        except Exception as e:
            logger.error(f"Agent processing failed: {e}", exc_info=True)
            elapsed_ms = int((time.time() - start_time) * 1000)
            return {
                "text": f"I encountered an error processing your request: {str(e)}",
                "tool_calls": tool_calls_made,
                "tool_results": tool_results,
                "citations": [],
                "debug": {
                    "rag_context_used": False,
                    "rag_chunk_ids": [],
                    "tools_called": [t['tool_name'] for t in tool_calls_made],
                    "iterations": 0,
                    "error": str(e),
                    "latency_ms": elapsed_ms
                }
            }
    
    async def _get_rag_context(
        self, 
        query: str, 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from RAG.
        
        Args:
            query: User query to search for
            top_k: Number of chunks to retrieve
            
        Returns:
            List of context blocks with chunk_id, text, source, similarity
        """
        try:
            # Generate query embedding
            query_embedding = await embedding_service.embed_query(query)
            
            # Vector similarity search
            search_results = await db.vector_search(query_embedding, top_k)
            
            # Deduplicate by chunk_id prefix (MMR-lite)
            seen_prefixes = set()
            context_blocks = []
            
            for result in search_results:
                chunk_id = result.get('chunk_id', '')
                # Extract base ID (before #) for deduplication
                base_id = chunk_id.split('#')[0] if '#' in chunk_id else chunk_id
                
                if base_id not in seen_prefixes:
                    context_blocks.append(result)
                    seen_prefixes.add(base_id)
                
                # Limit context to avoid token overflow
                if len(context_blocks) >= 4:
                    break
            
            logger.info(f"Retrieved {len(context_blocks)} RAG context blocks")
            return context_blocks
            
        except Exception as e:
            logger.error(f"Failed to retrieve RAG context: {e}")
            return []
    
    def _build_initial_messages(
        self, 
        query: str, 
        rag_context: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Build the initial message list with system prompt, chat history, and RAG context.
        
        Args:
            query: User's query
            rag_context: Retrieved RAG context blocks
            chat_history: Optional previous conversation messages
            
        Returns:
            List of messages for the LLM
        """
        # Build context string with citations
        context_parts = []
        for block in rag_context:
            chunk_id = block.get('chunk_id', 'unknown')
            text = block.get('text', '')
            context_parts.append(f"[{chunk_id}] {text}")
        
        context_str = "\n\n".join(context_parts) if context_parts else "No relevant context found in knowledge base."
        
        # Get current date for accurate scheduling
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_year = datetime.now().year
        
        # System prompt that enables both RAG and tool use
        system_prompt = f"""You are an intelligent AI assistant with access to both a knowledge base and action tools.

## IMPORTANT: Current Date Information
Today's date is: {current_date}
Current year is: {current_year}
When scheduling events, ALWAYS use the current year ({current_year}) or future dates. Never use past dates.

## Your Capabilities:

1. **Knowledge Base (RAG)**: You have access to retrieved context from our database. Use it to answer questions about policies, procedures, scheduling rules, and other information.

2. **Tools**: You can take real actions:
   - `create_calendar_event`: Schedule meetings on Google Calendar
   - `send_email`: Send emails via Gmail

## Retrieved Context from Knowledge Base:
{context_str}

## Instructions:

1. **For informational questions** (about policies, returns, shipping, scheduling rules, etc.):
   - Answer using the retrieved context above
   - Include citations in format [chunk_id] when referencing information
   - If context doesn't contain relevant info, say so

2. **For action requests** (schedule meeting, send email, etc.):
   - Use the appropriate tool
   - If RAG context contains relevant info (e.g., default meeting duration), USE IT
   - If missing required info (date, time, email), ASK the user
   - ALWAYS use the current year ({current_year}) for dates

3. **After executing a tool**:
   - Confirm the action with relevant details
   - Include any important information (event link, meeting time, etc.)

## Important Guidelines:
- Always check if RAG context is relevant before using it
- Include citations [chunk_id] when referencing knowledge base information
- For scheduling: Check context for default durations (e.g., "standard consultation = 30 minutes")
- Be concise, professional, and helpful
- Corporate email for sending: {settings.google_calendar_email}"""

        # Build messages list starting with system prompt
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add chat history if provided (for multi-turn conversations)
        if chat_history:
            for msg in chat_history:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        # Add current user query
        messages.append({"role": "user", "content": query})
        
        return messages
    
    async def _call_llm_with_tools(
        self, 
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Call the LLM with tool definitions.
        
        Args:
            messages: Conversation messages
            
        Returns:
            Dict with 'content' and optional 'tool_calls'
        """
        if self.provider == "openai":
            return await self._call_openai(messages)
        elif self.provider == "anthropic":
            return await self._call_anthropic(messages)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def _call_openai(
        self, 
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Call OpenAI API with tool definitions.
        
        Args:
            messages: Conversation messages
            
        Returns:
            Dict with 'content' and optional 'tool_calls'
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            temperature=settings.temperature,
            max_tokens=1500
        )
        
        message = response.choices[0].message
        
        return {
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in (message.tool_calls or [])
            ] if message.tool_calls else None
        }
    
    async def _call_anthropic(
        self, 
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Call Anthropic API with tool definitions.
        
        Args:
            messages: Conversation messages
            
        Returns:
            Dict with 'content' and optional 'tool_calls'
        """
        # Extract system message
        system_content = ""
        user_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                user_messages.append(msg)
        
        # Convert tool definitions to Anthropic format
        anthropic_tools = [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "input_schema": t["function"]["parameters"]
            }
            for t in TOOL_DEFINITIONS
        ]
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            temperature=settings.temperature,
            system=system_content,
            messages=user_messages,
            tools=anthropic_tools
        )
        
        # Parse Anthropic response format
        tool_calls = []
        content = ""
        
        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input)
                    }
                })
        
        return {
            "content": content,
            "tool_calls": tool_calls if tool_calls else None
        }
    
    def _extract_citations(
        self, 
        text: str, 
        context_blocks: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract citations from the response text.
        
        Args:
            text: Generated response text
            context_blocks: RAG context blocks that were provided
            
        Returns:
            List of valid chunk_ids that were cited
        """
        import re
        
        # Find all citations in format [chunk_id]
        citation_pattern = r'\[([^\]]+)\]'
        found_citations = re.findall(citation_pattern, text)
        
        # Filter to only include valid chunk_ids from context
        valid_chunk_ids = {block['chunk_id'] for block in context_blocks}
        valid_citations = [
            cite for cite in found_citations 
            if cite in valid_chunk_ids
        ]
        
        # Remove duplicates while preserving order
        unique_citations = []
        for cite in valid_citations:
            if cite not in unique_citations:
                unique_citations.append(cite)
        
        return unique_citations


# Global agent service instance
agent_service = AgentService()

