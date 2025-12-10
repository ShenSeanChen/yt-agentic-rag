# Directory: yt-agentic-rag/app/agents/tools/email_tool.py

"""
Email Tool - Send Emails via Gmail API.

This tool allows the agent to send emails with:
- Plain text body
- Custom subject lines
- Sent from the configured corporate email

Uses Gmail API with service account authentication and
domain-wide delegation to send emails on behalf of users.
"""

import logging
import base64
from typing import Dict, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from .base import BaseTool

logger = logging.getLogger(__name__)


class EmailTool(BaseTool):
    """
    Tool for sending emails via Gmail API.
    
    Uses Gmail API with service account credentials.
    The service account must have domain-wide delegation enabled
    to send emails on behalf of the configured email address.
    
    Required setup:
    1. Create a Google Cloud project
    2. Enable Gmail API
    3. Create a service account with domain-wide delegation
    4. Download the service account JSON key
    5. Grant the service account Gmail send access in Google Workspace Admin
       (scope: https://www.googleapis.com/auth/gmail.send)
    """
    
    def __init__(self):
        """Initialize the Email tool."""
        self._service = None
        self._initialized = False
    
    @property
    def name(self) -> str:
        """Tool name matching TOOL_DEFINITIONS."""
        return "send_email"
    
    @property
    def description(self) -> str:
        """Human-readable description."""
        return "Send an email via Gmail"
    
    def _get_service(self):
        """
        Lazily initialize and return the Gmail service.
        
        Returns:
            Gmail API service instance
            
        Raises:
            Exception: If credentials are not configured or invalid
        """
        if not self._initialized:
            try:
                # Import here to avoid issues if google packages not installed
                from google.oauth2 import service_account
                from googleapiclient.discovery import build
                
                # Import settings lazily to avoid circular imports
                from ...config.settings import get_settings
                settings = get_settings()
                
                # Load service account credentials
                credentials = service_account.Credentials.from_service_account_file(
                    settings.google_service_account_path,
                    scopes=['https://www.googleapis.com/auth/gmail.send']
                )
                
                # Delegate to the corporate email for domain-wide delegation
                delegated_credentials = credentials.with_subject(
                    settings.google_calendar_email  # Same email for both services
                )
                
                # Build the Gmail service
                self._service = build(
                    'gmail', 
                    'v1', 
                    credentials=delegated_credentials
                )
                self._initialized = True
                logger.info(
                    f"Gmail service initialized for "
                    f"{settings.google_calendar_email}"
                )
                
            except FileNotFoundError as e:
                logger.error(
                    f"Service account file not found: {e}. "
                    "Please ensure credentials/service_account.json exists."
                )
                raise
            except Exception as e:
                logger.error(f"Failed to initialize Gmail service: {e}")
                raise
        
        return self._service
    
    async def execute(
        self,
        to: str,
        subject: str,
        body: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send an email.
        
        Args:
            to: Recipient email address
            subject: Email subject line
            body: Email body content (plain text)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Dict with success status and message details or error
        """
        # Validate required parameters
        is_valid, missing = self.validate_params(
            required=['to', 'subject', 'body'],
            provided={'to': to, 'subject': subject, 'body': body}
        )
        
        if not is_valid:
            return self._error_response(
                f"Missing required parameters: {', '.join(missing)}"
            )
        
        try:
            # Import here to handle case where google packages not installed
            from googleapiclient.errors import HttpError
            from ...config.settings import get_settings
            settings = get_settings()
            
            service = self._get_service()
            
            # Create email message
            message = MIMEMultipart()
            message['to'] = to
            message['from'] = settings.google_calendar_email
            message['subject'] = subject
            
            # Attach the body as plain text
            message.attach(MIMEText(body, 'plain'))
            
            # Encode message to base64 for Gmail API
            raw_message = base64.urlsafe_b64encode(
                message.as_bytes()
            ).decode('utf-8')
            
            # Send the email
            sent_message = service.users().messages().send(
                userId='me',
                body={'raw': raw_message}
            ).execute()
            
            logger.info(
                f"Email sent successfully: "
                f"ID={sent_message.get('id')}, To='{to}', Subject='{subject}'"
            )
            
            return self._success_response({
                "message_id": sent_message.get('id'),
                "thread_id": sent_message.get('threadId'),
                "to": to,
                "subject": subject,
                "from": settings.google_calendar_email,
                "labels": sent_message.get('labelIds', [])
            })
            
        except ImportError as e:
            return self._error_response(
                f"Google API packages not installed. "
                f"Run: pip install google-api-python-client google-auth. "
                f"Error: {str(e)}"
            )
        except HttpError as e:
            logger.error(f"Gmail API error: {e}")
            return self._error_response(
                f"Gmail API error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Failed to send email: {e}", exc_info=True)
            return self._error_response(
                f"Failed to send email: {str(e)}"
            )


# Global tool instance
email_tool = EmailTool()

