"""Tests for LLM-native PDF integration."""

import base64
import pytest
from unittest.mock import patch, MagicMock, mock_open
from benchwarmer.agents.implementation import ImplementationAgent


class TestPDFNativeDocs:
    def test_pdf_document_block_creation(self):
        agent = ImplementationAgent(api_key="dummy")
        
        pdf_content = b"%PDF-1.4 mock content"
        encoded_pdf = base64.b64encode(pdf_content).decode("utf-8")
        
        # Mock file opening and API client
        with patch("builtins.open", mock_open(read_data=pdf_content)):
            with patch.object(agent.client.messages, "create") as mock_create:
                # Mock a successful response to avoid errors
                mock_create.return_value.content = [MagicMock(text="```python\nclass A(AlgorithmWrapper): pass\n```")]
                
                agent.generate(
                    description="impl this",
                    problem_class="max_cut",
                    pdf_paths=["paper1.pdf", "paper2.pdf"]
                )
                
                # Verify the API call structure
                call_args = mock_create.call_args[1]
                messages = call_args["messages"]
                user_content = messages[0]["content"]
                
                # Should detect document blocks
                doc_blocks = [b for b in user_content if b.get("type") == "document"]
                assert len(doc_blocks) == 2
                assert doc_blocks[0]["source"]["media_type"] == "application/pdf"
                assert doc_blocks[1]["source"]["media_type"] == "application/pdf"
                
                # Should also have text prompt
                text_block = next((b for b in user_content if b.get("type") == "text"), None)
                assert text_block is not None
                assert "Implement this algorithm" in text_block["text"]
