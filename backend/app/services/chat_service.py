"""Chat service that integrates character personalities with LLM services.

Following VCPToolBox pattern:
1. Stream LLM response
2. Detect tool calls in response
3. Execute tools
4. Call LLM again with tool results
5. Repeat until no more tool calls
"""

from typing import List, Dict, Optional, AsyncGenerator, Any, Tuple
import random
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

from app.services.llm import LLM
from app.services.character_service import CharacterStorageService
from app.models.character import UserCharacterPreference
from app.schemas.message import (
    ChatRequest,
    ChatResponse,
    MessageContext
)

# Tool calling imports
from plugins.tool_call_parser import ToolCallParser
from plugins.tool_executor import ToolExecutor


class ChatService:
    """
    Enhanced chat service that integrates character personalities with LLM services.

    Tool calling flow (VCPToolBox pattern):
    1. Generate LLM response (streaming or non-streaming)
    2. Parse response for tool calls <<<[TOOL_REQUEST]>>>...<<<[END_TOOL_REQUEST]>>>
    3. If tools found: execute them and call LLM again with results
    4. Repeat until no more tool calls or max iterations reached
    """

    def __init__(
        self,
        llm: LLM,
        character_service: CharacterStorageService,
        plugin_manager=None
    ):
        """
        Initialize chat service.

        Args:
            llm: LLM instance to use for generating responses
            character_service: Character service for managing personalities
            plugin_manager: Optional PluginManager for tool calling
        """
        self.llm = llm
        self.character_service = character_service

        # Initialize tool system if plugin_manager provided
        self.plugin_manager = plugin_manager
        if plugin_manager:
            self.tool_parser = ToolCallParser()
            self.tool_executor = ToolExecutor(plugin_manager)
            self.max_tool_iterations = 5  # Prevent infinite loops
        else:
            self.tool_parser = None
            self.tool_executor = None
            self.max_tool_iterations = 0

    def _build_message_context(
        self,
        request: ChatRequest
    ) -> Optional[MessageContext]:
        """
        Build message context based on request metadata.

        Args:
            request: Chat request

        Returns:
            MessageContext or None
        """
        # Get character to check behavior parameters
        character = self.character_service.get_character(request.character_id)

        if not character:
            return None

        # Default behavior parameters (simplified for file system storage)
        character_state = {"proactivity_level": 0.5, "argument_avoidance_threshold": 0.7}
        initiate_topic = random.random() < 0.5

        return MessageContext(
            character_state=character_state,
            initiate_topic=initiate_topic
        )

    async def chat(
        self,
        request: ChatRequest,
        user_preferences: Optional[UserCharacterPreference] = None,
        user_id: str = "user_default"
    ) -> ChatResponse:
        """
        Generate a character-aware response.

        This method collects the full response from chat_stream,
        supporting tool calling through the streaming implementation.
        """
        # Collect all chunks from stream
        full_response = ""
        async for chunk in self.chat_stream(request, user_preferences, user_id):
            full_response += chunk

        # Build response object
        message_context = self._build_message_context(request)
        return ChatResponse(
            message=full_response,
            character_id=request.character_id,
            context_used=message_context.dict() if message_context else None,
            timestamp=datetime.now()
        )

    async def chat_stream(
        self,
        request: ChatRequest,
        user_preferences: Optional[UserCharacterPreference] = None,
        user_id: str = "user_default"
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming character-aware response with tool calling support.

        pattern:
        1. Stream LLM response
        2. Check for tool calls <<<[TOOL_REQUEST]>>>...<<<[END_TOOL_REQUEST]>>>
        3. If tools found: execute and stream new response with results
        4. Repeat until no more tool calls

        Yields:
            Response chunks
        """
        # Build initial messages
        messages = await self._build_messages(request, user_preferences, user_id)

        # Tool calling loop
        iteration = 0
        max_iterations = self.max_tool_iterations if self.plugin_manager else 1

        while iteration < max_iterations:
            iteration += 1

            # Stream response and check for tool calls
            response_chunks = []  # All chunks including tool markers (for history)
            tool_calls = []
            current_tool_content = ""
            in_tool_call = False

            for chunk in self.llm.generate_response_stream(messages):
                response_chunks.append(chunk)

                # Track tool call state
                if "<<<[TOOL_REQUEST]>>>" in chunk:
                    in_tool_call = True
                    current_tool_content = chunk

                if in_tool_call:
                    current_tool_content += chunk
                    if "<<<[END_TOOL_REQUEST]>>>" in current_tool_content:
                        in_tool_call = False
                        # Parse tool call after complete marker
                        if self.tool_parser:
                            parsed_calls = self.tool_parser.parse(current_tool_content)
                            tool_calls.extend(parsed_calls)
                        current_tool_content = ""

                # Yield ALL content (including tool call markers) - VCPToolBox pattern
                yield chunk

            # After stream completes, check for tool calls in full response
            # (in case markers spanned multiple chunks)
            full_response = "".join(response_chunks)
            if "<<<[TOOL_REQUEST]>>>" in full_response and not tool_calls:
                # Try parsing the full response
                if self.tool_parser:
                    tool_calls = self.tool_parser.parse(full_response)

            # Check if we found any tool calls
            if not tool_calls:
                # No tool calls - we're done
                break

            # Log detected tool calls
            logger.info(f"[Tool Call] Executing {len(tool_calls)} tool(s): {[tc.name for tc in tool_calls]}")

            # Log tool call arguments (placeholders already pre-replaced in _build_messages)
            for tc in tool_calls:
                if tc.args:
                    logger.info(f"[Tool Call] {tc.name} args={tc.args}")

            # Execute tools
            if self.tool_executor:
                execution_results = await self.tool_executor.execute_all(tool_calls)

                # Log execution results
                for i, result in enumerate(execution_results):
                    tool_name = result.get('tool_name', 'Unknown')
                    if result.get('success'):
                        content = str(result.get('content', ''))[:100]
                        logger.info(f"[Tool Call] [{i+1}/{len(tool_calls)}] {tool_name} - SUCCESS: {content}...")
                    else:
                        error = result.get('error', 'Unknown error')
                        logger.error(f"[Tool Call] [{i+1}/{len(tool_calls)}] {tool_name} - FAILED: {error}")

                tool_summary = self.format_tool_results(execution_results)

                # Save FULL response (including tool markers) - VCPToolBox pattern
                full_response = "".join(response_chunks)
                messages.append({"role": "assistant", "content": full_response})
                messages.append({
                    "role": "user",
                    "content": f"<!-- VCP_TOOL_PAYLOAD -->\n{tool_summary}"
                })
                # Continue loop to generate new response with tool results
            else:
                break

    async def _build_messages(
        self,
        request: ChatRequest,
        user_preferences: Optional[UserCharacterPreference],
        user_id: str
    ) -> List[Dict]:
        """
        Build messages list for LLM call.

        Returns:
            List of message dicts ready for LLM
        """
        # Build message context
        message_context = self._build_message_context(request)

        # Generate system prompt
        system_prompt = self.character_service.get_prompt(request.character_id)
        if not system_prompt:
            raise ValueError(f"Character not found: {request.character_id}")

        # Add tool descriptions if plugin manager available
        if self.plugin_manager:
            # Get all tools descriptions
            tools_description = self.plugin_manager.get_all_tools_description()

            # Add tools description to system prompt
            system_prompt = f"{system_prompt}\n\n{tools_description}"

            logger.info(f"[Tool Call] Added tools description to system prompt")

        # Build messages list
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history if provided
        if request.conversation_history:
            messages.extend(request.conversation_history)

        # Add current message
        messages.append({"role": "user", "content": request.message})

        return messages

    def _extract_response_without_tool_calls(self, response: str) -> str:
        """
        Extract response content without tool call markers.

        Args:
            response: Response that may contain tool calls

        Returns:
            Clean response without tool call markers
        """
        pattern = r'<<<\[TOOL_REQUEST\]>>>.*?<<<\[END_TOOL_REQUEST\]>>>'
        return re.sub(pattern, '', response, flags=re.DOTALL).strip()

    def format_tool_results(self, execution_results: List[Dict]) -> str:
        """
        Format tool execution results for injection into conversation.

        Args:
            execution_results: List of execution result dicts

        Returns:
            Formatted summary string
        """
        summary_parts = ["[[VCP调用结果信息汇总"]

        for result in execution_results:
            tool_name = result.get('tool_name', 'Unknown')
            if result.get('success'):
                content = str(result.get('content', ''))
                # Truncate long content
                if len(content) > 1000:
                    content = content[:1000] + "..."
                summary_parts.append(f"- 工具名称: {tool_name}")
                summary_parts.append(f"- 执行状态: success")
                summary_parts.append(f"- 返回内容: {content}")
            else:
                error_msg = result.get('error', '未知错误')
                summary_parts.append(f"- 工具名称: {tool_name}")
                summary_parts.append(f"- 执行状态: failed")
                summary_parts.append(f"- 错误信息: {error_msg}")

        summary_parts.append("VCP调用结果结束]]")

        result = "\n".join(summary_parts)

        # 添加提示，引导 AI 生成自然的响应
        return f"""{result}

请根据以上工具调用结果，继续与用户对话。如果工具执行成功，请用自然的方式告知用户操作已完成，并继续对话。不要重复工具调用的技术细节。"""
