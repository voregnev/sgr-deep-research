import json
import logging
import uuid
from typing import Type

from sgr_deep_research.core.agents.base_agent import BaseAgent
from sgr_deep_research.core.tools import (
    AgentCompletionTool,
    BaseTool,
    ClarificationTool,
    CreateReportTool,
    NextStepToolsBuilder,
    NextStepToolStub,
    ReasoningTool,
    WebSearchTool,
    research_agent_tools,
    system_agent_tools,
)
from sgr_deep_research.settings import get_config

logging.basicConfig(
    level=logging.INFO,
    encoding="utf-8",
    format="%(asctime)s - %(name)s - %(lineno)d - %(levelname)s -  - %(message)s",
    handlers=[logging.StreamHandler()],
)

config = get_config()
logger = logging.getLogger(__name__)


class SGRResearchAgent(BaseAgent):
    """Agent for deep research tasks using SGR framework."""

    def __init__(
        self,
        task: str,
        toolkit: list[Type[BaseTool]] | None = None,
        max_clarifications: int = 3,
        max_iterations: int = None,
        max_searches: int = None,
    ):
        # Use config values if not provided
        if max_iterations is None:
            max_iterations = config.execution.max_steps
        if max_searches is None:
            max_searches = min(config.execution.max_steps, 8)  # Limit searches to reasonable number
        
        super().__init__(
            task=task,
            toolkit=toolkit,
            max_clarifications=max_clarifications,
            max_iterations=max_iterations,
        )

        self.id = f"sgr_agent_{uuid.uuid4()}"

        self.toolkit = [
            *system_agent_tools,
            *research_agent_tools,
            *(toolkit or []),
        ]
        self.toolkit.remove(ReasoningTool)  # we use our own reasoning scheme
        self.max_searches = max_searches

    async def _prepare_tools(self) -> Type[NextStepToolStub]:
        """Prepare tool classes with current context limits."""
        tools = set(self.toolkit)
        
        # Force report creation if we're near limits or have enough data
        if (self._context.iteration >= self.max_iterations - 1 or 
            self._context.searches_used >= self.max_searches or
            len(self._context.sources) >= 3):
            tools = {
                CreateReportTool,
                AgentCompletionTool,
            }
            logger.info("Forcing report creation due to limits or sufficient data")
        else:
            if self._context.clarifications_used >= self.max_clarifications:
                tools -= {
                    ClarificationTool,
                }
            if self._context.searches_used >= self.max_searches:
                tools -= {
                    WebSearchTool,
                }
        
        logger.info(f"Available tools: {[tool.tool_name for tool in tools if hasattr(tool, 'tool_name')]}")
        return NextStepToolsBuilder.build_NextStepTools(list(tools))

    async def _reasoning_phase(self) -> NextStepToolStub:
        async with self.openai_client.chat.completions.stream(
            model=config.openai.model,
            messages=await self._prepare_context(),
            max_tokens=config.openai.max_tokens,
            temperature=config.openai.temperature,
        ) as stream:
            async for event in stream:
                if event.type == "chunk":
                    content = event.chunk.choices[0].delta.content
                    self.streaming_generator.add_chunk(content)
        # Parse JSON response manually since we're using json_object format
        final_completion = await stream.get_final_completion()
        response_content = final_completion.choices[0].message.content
        
        logger.info(f"Model response content: '{response_content}'")
        
        if not response_content:
            raise ValueError("Empty response from model")
        
        # Try to extract JSON from response if it's wrapped in text
        json_content = response_content.strip()
        
        # Remove markdown code blocks
        if json_content.startswith('```json'):
            json_content = json_content[7:]
        if json_content.startswith('```'):
            json_content = json_content[3:]
        if json_content.endswith('```'):
            json_content = json_content[:-3]
        json_content = json_content.strip()
        
        # Try to find JSON object in the text
        import re
        json_match = re.search(r'\{.*\}', json_content, re.DOTALL)
        if json_match:
            json_content = json_match.group(0)
        
        try:
            response_json = json.loads(json_content)
            # Store the original JSON for later use
            self._original_response_json = response_json
            # Create NextStepToolStub instance from JSON
            reasoning = NextStepToolStub.model_validate(response_json)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Raw content: '{json_content}'")
            logger.error(f"Original content: '{response_content}'")
            raise ValueError(f"Failed to parse model response as JSON: {e}")
        # we are not fully sure if it should be in conversation or not. Looks like not necessary data
        # self.conversation.append({"role": "assistant", "content": reasoning.model_dump_json(exclude={"function"})})
        self._log_reasoning(reasoning)
        return reasoning

    async def _select_action_phase(self, reasoning: NextStepToolStub) -> BaseTool:
        function_data = reasoning.function
        logger.info(f"Function data type: {type(function_data)}")
        logger.info(f"Function data: {function_data}")
        if isinstance(function_data, dict):
            # Convert dict to actual tool instance
            tool_name = function_data.get("tool_name_discriminator")
            if not tool_name:
                raise ValueError("No tool_name_discriminator found in function data")
            
            # Find the tool class by name
            tool_class = None
            for tool in self.toolkit:
                if hasattr(tool, 'tool_name') and tool.tool_name == tool_name:
                    tool_class = tool
                    break
            
            if not tool_class:
                raise ValueError(f"Tool class not found for name: {tool_name}")
            
            # Create tool instance from the data
            try:
                logger.info(f"Creating tool instance for {tool_name} with data: {function_data}")
                tool = tool_class.model_validate(function_data)
                logger.info(f"Created tool instance: {type(tool)} - {tool}")
            except Exception as e:
                logger.error(f"Failed to create tool instance: {e}")
                raise ValueError(f"Failed to create tool instance: {e}")
        else:
            # function_data is a BaseTool instance, but we need to extract the actual tool data
            # The issue is that Pydantic creates a BaseTool instance instead of the specific tool
            # Let's use the original JSON response that we saved
            logger.info(f"Function data is BaseTool instance, using original JSON response")
            
            try:
                # Use the original JSON response that we saved in _reasoning_phase
                if not hasattr(self, '_original_response_json'):
                    raise ValueError("No original response JSON found")
                
                function_data_dict = self._original_response_json.get("function", {})
                tool_name = function_data_dict.get("tool_name_discriminator")
                
                if not tool_name:
                    raise ValueError("No tool_name_discriminator found in original response JSON")
                
                logger.info(f"Tool name from original JSON: {tool_name}")
                
                # Find the tool class by name
                tool_class = None
                for tool in self.toolkit:
                    if hasattr(tool, 'tool_name') and tool.tool_name == tool_name:
                        tool_class = tool
                        break
                
                if not tool_class:
                    raise ValueError(f"Tool class not found for name: {tool_name}")
                
                # Create tool instance from the function data
                try:
                    logger.info(f"Creating tool instance for {tool_name} with data: {function_data_dict}")
                    
                    # Handle field mapping for different tools
                    if tool_name == "websearchtool":
                        # Handle arguments object if present
                        if "arguments" in function_data_dict and isinstance(function_data_dict["arguments"], dict):
                            # Extract data from arguments object
                            arguments = function_data_dict["arguments"]
                            function_data_dict.update(arguments)
                            del function_data_dict["arguments"]
                        
                        # Map queries array to single query string and add reasoning
                        queries_key = None
                        if "queries" in function_data_dict and isinstance(function_data_dict["queries"], list):
                            queries_key = "queries"
                        elif "search_queries" in function_data_dict and isinstance(function_data_dict["search_queries"], list):
                            queries_key = "search_queries"
                        
                        if queries_key:
                            # Use the first query as the main query
                            function_data_dict["query"] = function_data_dict[queries_key][0]
                            del function_data_dict[queries_key]
                        
                        # Add reasoning if missing
                        if "reasoning" not in function_data_dict:
                            function_data_dict["reasoning"] = f"Searching for {function_data_dict.get('query', 'information')}"
                    
                    elif tool_name == "clarificationtool":
                        # Handle questions field for ClarificationTool
                        if "questions" in function_data_dict and isinstance(function_data_dict["questions"], list):
                            # Convert questions to unclear_terms
                            function_data_dict["unclear_terms"] = function_data_dict["questions"]
                            del function_data_dict["questions"]
                        
                        # Add missing required fields
                        if "reasoning" not in function_data_dict:
                            function_data_dict["reasoning"] = "Need clarification on user requirements"
                        
                        if "unclear_terms" not in function_data_dict:
                            function_data_dict["unclear_terms"] = ["User requirements"]
                        
                        if "assumptions" not in function_data_dict:
                            function_data_dict["assumptions"] = ["User may need more specific information", "Additional context required"]
                        
                        if "questions" not in function_data_dict:
                            function_data_dict["questions"] = ["Could you provide more details?", "What specific aspect interests you?", "Any particular focus area?"]
                    
                    elif tool_name == "adaptplantool":
                        # Handle missing required fields for AdaptPlanTool
                        if "reasoning" not in function_data_dict:
                            function_data_dict["reasoning"] = "Adapting research plan based on new findings"
                        
                        if "original_goal" not in function_data_dict:
                            function_data_dict["original_goal"] = "Original research objective"
                        
                        if "new_goal" not in function_data_dict:
                            function_data_dict["new_goal"] = "Updated research objective"
                        
                        if "plan_changes" not in function_data_dict:
                            function_data_dict["plan_changes"] = ["Plan adaptation based on new data"]
                        
                        if "next_steps" not in function_data_dict:
                            function_data_dict["next_steps"] = ["Continue with adapted plan", "Monitor progress"]
                    
                    elif tool_name == "generateplantool":
                        # Handle missing required fields for GeneratePlanTool
                        if "reasoning" not in function_data_dict:
                            function_data_dict["reasoning"] = "Generating research plan"
                        
                        if "research_goal" not in function_data_dict:
                            function_data_dict["research_goal"] = "Research objective"
                        
                        if "planned_steps" not in function_data_dict:
                            function_data_dict["planned_steps"] = ["Step 1", "Step 2", "Step 3"]
                        
                        if "search_strategies" not in function_data_dict:
                            function_data_dict["search_strategies"] = ["Web search", "Official sources"]
                    
                    elif tool_name == "agentcompletiontool":
                        # Handle missing required fields for AgentCompletionTool
                        if "reasoning" not in function_data_dict:
                            function_data_dict["reasoning"] = "Task completion reasoning"
                        
                        if "completed_steps" not in function_data_dict:
                            function_data_dict["completed_steps"] = ["Research completed"]
                        
                        if "status" not in function_data_dict:
                            function_data_dict["status"] = "completed"
                    
                    elif tool_name == "createreporttool":
                        # Handle missing required fields for CreateReportTool
                        if "reasoning" not in function_data_dict:
                            function_data_dict["reasoning"] = "Creating comprehensive research report"
                        
                        if "title" not in function_data_dict:
                            function_data_dict["title"] = "Research Report"
                        
                        if "content" not in function_data_dict:
                            function_data_dict["content"] = "Research findings and analysis"
                        
                        if "confidence" not in function_data_dict:
                            function_data_dict["confidence"] = "medium"
                        
                        if "user_request_language_reference" not in function_data_dict:
                            function_data_dict["user_request_language_reference"] = "English"
                    
                    elif tool_name == "reasoningtool":
                        # Handle missing required fields for ReasoningTool
                        if "reasoning_steps" not in function_data_dict:
                            function_data_dict["reasoning_steps"] = ["Step 1", "Step 2", "Step 3"]
                        else:
                            # Ensure reasoning_steps doesn't exceed max length of 4
                            if len(function_data_dict["reasoning_steps"]) > 4:
                                function_data_dict["reasoning_steps"] = function_data_dict["reasoning_steps"][:4]
                        
                        if "current_situation" not in function_data_dict:
                            function_data_dict["current_situation"] = "Current research situation"
                        
                        if "plan_status" not in function_data_dict:
                            function_data_dict["plan_status"] = "Plan status"
                        
                        if "remaining_steps" not in function_data_dict:
                            function_data_dict["remaining_steps"] = ["Step 1", "Step 2", "Step 3"]
                        else:
                            # Ensure remaining_steps doesn't exceed max length of 3
                            if len(function_data_dict["remaining_steps"]) > 3:
                                function_data_dict["remaining_steps"] = function_data_dict["remaining_steps"][:3]
                        
                        if "task_completed" not in function_data_dict:
                            function_data_dict["task_completed"] = False
                    
                    # Apply length constraints to all list fields that might exceed limits
                    for field_name, max_length in [
                        ("reasoning_steps", 4),
                        ("remaining_steps", 3),
                        ("planned_steps", 4),
                        ("search_strategies", 3),
                        ("plan_changes", 3),
                        ("next_steps", 4),
                        ("completed_steps", 5),
                        ("unclear_terms", 5),
                        ("assumptions", 4),
                        ("questions", 5)
                    ]:
                        if field_name in function_data_dict and isinstance(function_data_dict[field_name], list):
                            if len(function_data_dict[field_name]) > max_length:
                                function_data_dict[field_name] = function_data_dict[field_name][:max_length]
                    
                    tool = tool_class.model_validate(function_data_dict)
                    logger.info(f"Created tool instance: {type(tool)} - {tool}")
                except Exception as e:
                    logger.error(f"Failed to create tool instance: {e}")
                    raise ValueError(f"Failed to create tool instance: {e}")
                    
            except Exception as e:
                logger.error(f"Failed to extract tool data from original JSON: {e}")
                raise ValueError(f"Failed to extract tool data from original JSON: {e}")
        self.conversation.append(
            {
                "role": "assistant",
                "content": reasoning.remaining_steps[0] if reasoning.remaining_steps else "Completing",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": f"{self._context.iteration}-action",
                        "function": {
                            "name": tool.tool_name,
                            "arguments": tool.model_dump_json(),
                        },
                    }
                ],
            }
        )
        self.streaming_generator.add_tool_call(
            f"{self._context.iteration}-action", tool.tool_name, tool.model_dump_json()
        )
        return tool

    async def _action_phase(self, tool: BaseTool) -> str:
        result = tool(self._context)
        self.conversation.append(
            {"role": "tool", "content": result, "tool_call_id": f"{self._context.iteration}-action"}
        )
        self.streaming_generator.add_chunk(f"{result}\n")
        self._log_tool_execution(tool, result)
        return result
