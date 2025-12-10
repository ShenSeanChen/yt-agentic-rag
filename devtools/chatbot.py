#!/usr/bin/env python3
# Copyright 2024
# Directory: yt-agentic-rag/devtools/chatbot.py

"""
Terminal Chatbot Client for Agentic RAG API.

A beautiful terminal-based chatbot to interact with the Agentic RAG backend API.
Enhanced with rich styling, colors, and modern UI elements.

Usage:
    python devtools/chatbot.py [base_url]
    
Example:
    python devtools/chatbot.py http://localhost:8000
"""

import requests
import json
import sys
import time
import os
from datetime import datetime
from typing import Dict, Any

# Rich library for beautiful terminal output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.prompt import Prompt
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich import box
    from rich.align import Align
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("📦 Installing rich for beautiful terminal UI...")
    os.system("pip install rich")
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        from rich.prompt import Prompt
        from rich.table import Table
        from rich.progress import Progress, SpinnerColumn, TextColumn
        from rich.markdown import Markdown
        from rich.syntax import Syntax
        from rich import box
        from rich.align import Align
        RICH_AVAILABLE = True
    except ImportError:
        RICH_AVAILABLE = False


class RAGChatbot:
    """Beautiful terminal chatbot for interacting with RAG backend."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the chatbot.
        
        Args:
            base_url (str): Base URL of the RAG backend API
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.console = Console() if RICH_AVAILABLE else None
        self.conversation_count = 0
        
    def check_health(self) -> bool:
        """
        Check if the backend is healthy.
        
        Returns:
            bool: True if backend is healthy, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/healthz", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def seed_documents(self) -> Dict[str, Any]:
        """
        Seed the backend with default documents.
        
        Returns:
            Dict[str, Any]: Response from the seed endpoint
        """
        try:
            response = self.session.post(f"{self.base_url}/seed", timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def ask_question(self, query: str, top_k: int = 6) -> Dict[str, Any]:
        """
        Ask a question to the RAG backend.
        
        Args:
            query (str): The question to ask
            top_k (int): Number of top results to retrieve
            
        Returns:
            Dict[str, Any]: Response from the answer endpoint
        """
        try:
            payload = {"query": query, "top_k": top_k}
            response = self.session.post(
                f"{self.base_url}/answer",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def print_welcome(self):
        """Print beautiful welcome message."""
        if not RICH_AVAILABLE:
            self._print_welcome_fallback()
            return
            
        # Create animated title
        title = Text()
        title.append("🤖 ", style="bold blue")
        title.append("RAG AI Assistant", style="bold magenta")
        title.append(" ✨", style="bold yellow")
        
        subtitle = Text()
        subtitle.append("Powered by ", style="dim")
        subtitle.append("OpenAI GPT-4o", style="bold green")
        subtitle.append(" + ", style="dim")
        subtitle.append("Vector Search", style="bold cyan")
        
        welcome_panel = Panel(
            Align.center(title + "\n" + subtitle),
            box=box.DOUBLE,
            border_style="bright_blue",
            padding=(1, 2)
        )
        
        self.console.print(welcome_panel)
        
        # Commands table
        commands_table = Table(
            title="Available Commands",
            box=box.ROUNDED,
            border_style="bright_green",
            title_style="bold green"
        )
        commands_table.add_column("Command", style="bold cyan", no_wrap=True)
        commands_table.add_column("Description", style="white")
        
        commands_table.add_row("/help", "Show this help message")
        commands_table.add_row("/health", "Check backend health status")
        commands_table.add_row("/seed", "Load documents into vector database")
        commands_table.add_row("/stats", "Show conversation statistics")
        commands_table.add_row("/clear", "Clear the screen")
        commands_table.add_row("/quit", "Exit the chatbot")
        
        self.console.print(commands_table)
        
        info_text = Text()
        info_text.append("💡 ", style="bold yellow")
        info_text.append("Just type your question to get started! ", style="white")
        info_text.append("Ask about policies, returns, shipping, and more.", style="dim")
        
        self.console.print(Panel(info_text, border_style="yellow"))
    
    def _print_welcome_fallback(self):
        """Fallback welcome message without rich."""
        print("\n" + "="*70)
        print("🤖 RAG AI Assistant - Terminal Chat ✨")
        print("="*70)
        print("Ask me anything about our policies and services!")
        print("\nCommands:")
        print("  /help    - Show this help message")
        print("  /health  - Check backend health")
        print("  /seed    - Seed documents into vector database")
        print("  /stats   - Show conversation statistics")
        print("  /clear   - Clear the screen")
        print("  /quit    - Exit the chatbot")
        print("="*70 + "\n")
    
    def print_help(self):
        """Print beautiful help message."""
        if not RICH_AVAILABLE:
            self._print_help_fallback()
            return
            
        help_table = Table(
            title="🆘 Help & Commands",
            box=box.HEAVY,
            border_style="bright_magenta",
            title_style="bold magenta"
        )
        help_table.add_column("Command", style="bold cyan", no_wrap=True, width=12)
        help_table.add_column("Description", style="white")
        help_table.add_column("Example", style="dim italic")
        
        help_table.add_row("/help", "Show this help message", "/help")
        help_table.add_row("/health", "Check if backend is running", "/health")
        help_table.add_row("/seed", "Load default documents", "/seed")
        help_table.add_row("/stats", "Show chat statistics", "/stats")
        help_table.add_row("/clear", "Clear the terminal screen", "/clear")
        help_table.add_row("/quit", "Exit the chatbot", "/quit")
        
        self.console.print(help_table)
        
        examples = """
## 💬 Example Questions:
- "Can I return shoes after 30 days?"
- "What's your shipping policy?"
- "How do I exchange a defective item?"
- "What items cannot be returned?"
        """
        
        self.console.print(Panel(Markdown(examples), border_style="green", title="Examples", title_align="left"))
    
    def _print_help_fallback(self):
        """Fallback help message without rich."""
        print("\n📋 Available Commands:")
        print("  /help    - Show this help message")
        print("  /health  - Check if the backend is running")
        print("  /seed    - Load default documents into the database")
        print("  /stats   - Show conversation statistics")
        print("  /clear   - Clear the screen")
        print("  /quit    - Exit the chatbot")
        print("\n💡 Just type your question to get started!")
    
    def show_stats(self):
        """Show conversation statistics."""
        if not RICH_AVAILABLE:
            print(f"📊 Conversations: {self.conversation_count}")
            return
            
        stats_table = Table(
            title="📊 Chat Statistics",
            box=box.SIMPLE,
            border_style="bright_yellow"
        )
        stats_table.add_column("Metric", style="bold")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Conversations", str(self.conversation_count))
        stats_table.add_row("Backend URL", self.base_url)
        stats_table.add_row("Session Start", datetime.now().strftime("%H:%M:%S"))
        
        self.console.print(stats_table)
    
    def format_response(self, response: Dict[str, Any]) -> None:
        """
        Format and display the response beautifully.
        
        Args:
            response (Dict[str, Any]): Response from the backend
        """
        if not RICH_AVAILABLE:
            self._format_response_fallback(response)
            return
            
        if "error" in response:
            error_panel = Panel(
                f"❌ {response['error']}",
                border_style="red",
                title="Error",
                title_align="left"
            )
            self.console.print(error_panel)
            return
        
        if "text" in response:
            # Main response
            response_text = Text()
            response_text.append("🤖 ", style="bold blue")
            response_text.append(response['text'], style="white")
            
            response_panel = Panel(
                response_text,
                border_style="blue",
                title="AI Assistant",
                title_align="left",
                padding=(1, 2)
            )
            self.console.print(response_panel)
            
            # Citations and metadata
            if "citations" in response and response["citations"]:
                citations_text = Text()
                citations_text.append("📚 Sources: ", style="bold green")
                citations_text.append(", ".join(response["citations"]), style="cyan")
                
                self.console.print(Panel(citations_text, border_style="green"))
            
            if "debug" in response and "latency_ms" in response["debug"]:
                debug_text = Text()
                debug_text.append("⏱️ ", style="yellow")
                debug_text.append(f"Response time: {response['debug']['latency_ms']}ms", style="dim")
                
                self.console.print(debug_text)
    
    def _format_response_fallback(self, response: Dict[str, Any]):
        """Fallback response formatting without rich."""
        if "error" in response:
            print(f"❌ Error: {response['error']}")
            return
        
        if "text" in response:
            print(f"🤖 {response['text']}\n")
            
            if "citations" in response and response["citations"]:
                print(f"📚 Sources: {', '.join(response['citations'])}")
            
            if "debug" in response and "latency_ms" in response["debug"]:
                print(f"⏱️  Response time: {response['debug']['latency_ms']}ms")
    
    def show_thinking_animation(self):
        """Show a thinking animation."""
        if not RICH_AVAILABLE:
            print("🤔 Thinking...")
            return
            
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task("🤔 Thinking...", total=None)
            time.sleep(0.5)  # Brief animation
    
    def run(self):
        """Run the interactive chatbot."""
        # Clear screen for better presentation
        if RICH_AVAILABLE:
            self.console.clear()
        else:
            os.system('clear' if os.name == 'posix' else 'cls')
            
        self.print_welcome()
        
        # Check health on startup
        if not self.check_health():
            if RICH_AVAILABLE:
                warning = Panel(
                    "❌ Backend appears to be down. Please start the server first.\n"
                    "Run: [bold cyan]uvicorn main:app --reload --port 8000[/bold cyan]",
                    border_style="red",
                    title="Warning",
                    title_align="left"
                )
                self.console.print(warning)
            else:
                print("❌ Warning: Backend appears to be down. Please start the server first.")
                print("   Run: uvicorn main:app --reload --port 8000\n")
        else:
            if RICH_AVAILABLE:
                success = Panel(
                    "✅ Backend is healthy and ready to chat!",
                    border_style="green",
                    title="Status",
                    title_align="left"
                )
                self.console.print(success)
            else:
                print("✅ Backend is healthy and ready!\n")
        
        while True:
            try:
                if RICH_AVAILABLE:
                    user_input = Prompt.ask("\n[bold green]You[/bold green]", console=self.console).strip()
                else:
                    user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == "/quit":
                    if RICH_AVAILABLE:
                        goodbye = Panel(
                            "👋 Thanks for chatting! Have a great day! ✨",
                            border_style="magenta",
                            title="Goodbye",
                            title_align="center"
                        )
                        self.console.print(goodbye)
                    else:
                        print("\n👋 Thanks for chatting! Goodbye!")
                    break
                    
                elif user_input.lower() == "/help":
                    self.print_help()
                    continue
                    
                elif user_input.lower() == "/health":
                    if self.check_health():
                        if RICH_AVAILABLE:
                            self.console.print("✅ [green]Backend is healthy![/green]")
                        else:
                            print("✅ Backend is healthy!")
                    else:
                        if RICH_AVAILABLE:
                            self.console.print("❌ [red]Backend is not responding.[/red]")
                        else:
                            print("❌ Backend is not responding.")
                    continue
                    
                elif user_input.lower() == "/seed":
                    if RICH_AVAILABLE:
                        with Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            console=self.console,
                        ) as progress:
                            task = progress.add_task("🌱 Seeding documents...", total=None)
                            result = self.seed_documents()
                    else:
                        print("🌱 Seeding documents...")
                        result = self.seed_documents()
                        
                    if "error" in result:
                        if RICH_AVAILABLE:
                            error_panel = Panel(
                                f"❌ Seeding failed: {result['error']}",
                                border_style="red",
                                title="Error"
                            )
                            self.console.print(error_panel)
                        else:
                            print(f"❌ Seeding failed: {result['error']}")
                    else:
                        if RICH_AVAILABLE:
                            success_panel = Panel(
                                f"✅ Successfully seeded {result.get('inserted', 0)} documents!",
                                border_style="green",
                                title="Success"
                            )
                            self.console.print(success_panel)
                        else:
                            print(f"✅ Successfully seeded {result.get('inserted', 0)} documents!")
                    continue
                    
                elif user_input.lower() == "/stats":
                    self.show_stats()
                    continue
                    
                elif user_input.lower() == "/clear":
                    if RICH_AVAILABLE:
                        self.console.clear()
                    else:
                        os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                
                # Regular question
                self.show_thinking_animation()
                response = self.ask_question(user_input)
                self.conversation_count += 1
                self.format_response(response)
                
            except KeyboardInterrupt:
                if RICH_AVAILABLE:
                    goodbye = Panel(
                        "👋 Thanks for chatting! Have a great day! ✨",
                        border_style="magenta",
                        title="Goodbye",
                        title_align="center"
                    )
                    self.console.print(goodbye)
                else:
                    print("\n\n👋 Thanks for chatting! Goodbye!")
                break
            except EOFError:
                if RICH_AVAILABLE:
                    goodbye = Panel(
                        "👋 Thanks for chatting! Have a great day! ✨",
                        border_style="magenta",
                        title="Goodbye",
                        title_align="center"
                    )
                    self.console.print(goodbye)
                else:
                    print("\n\n👋 Thanks for chatting! Goodbye!")
                break


if __name__ == "__main__":
    # Allow custom base URL via command line argument
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    chatbot = RAGChatbot(base_url)
    chatbot.run()