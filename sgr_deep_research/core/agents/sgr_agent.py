import json
import logging
import uuid
from datetime import datetime
from typing import Type

from sgr_deep_research.core.agents.base_agent import BaseAgent
from sgr_deep_research.core.models import AgentStatesEnum
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
from sgr_deep_research.core.tools.research import FinalReportTool
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
            
            # Always include CreateReportTool when we have sources
            if len(self._context.sources) > 0:
                tools.add(CreateReportTool)
                logger.info("Added CreateReportTool due to available sources")
        
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
        
        logger.info(f"🔍 Model response content length: {len(response_content) if response_content else 0}")
        logger.info(f"🔍 Model response content preview: '{response_content[:200] if response_content else 'None'}...'")
        
        if not response_content:
            logger.error("❌ Empty response from model")
            raise ValueError("Empty response from model")
        
        # Try to extract JSON from response if it's wrapped in text
        json_content = response_content.strip()
        logger.info(f"🔍 Initial json_content length: {len(json_content)}")
        logger.info(f"🔍 Initial json_content preview: '{json_content[:200]}...'")
        
        # Remove markdown code blocks
        original_json_content = json_content
        if json_content.startswith('```json'):
            json_content = json_content[7:]
            logger.info("🔍 Removed ```json prefix")
        if json_content.startswith('```'):
            json_content = json_content[3:]
            logger.info("🔍 Removed ``` prefix")
        if json_content.endswith('```'):
            json_content = json_content[:-3]
            logger.info("🔍 Removed ``` suffix")
        json_content = json_content.strip()
        
        logger.info(f"🔍 After markdown cleanup - json_content length: {len(json_content)}")
        logger.info(f"🔍 After markdown cleanup - json_content preview: '{json_content[:200]}...'")
        
        # Try to find JSON object in the text
        import re
        json_match = re.search(r'\{.*\}', json_content, re.DOTALL)
        if json_match:
            json_content = json_match.group(0)
            logger.info("🔍 Found JSON object using regex")
            logger.info(f"🔍 Extracted JSON length: {len(json_content)}")
            logger.info(f"🔍 Extracted JSON preview: '{json_content[:200]}...'")
        else:
            logger.warning("⚠️ No JSON object found using regex pattern")
        
        # Log the exact content that will be parsed
        logger.info(f"🔍 Final JSON content to parse:")
        logger.info(f"🔍 Length: {len(json_content)}")
        logger.info(f"🔍 Content: '{json_content}'")
        
        # Check for common JSON issues
        if json_content.count('{') != json_content.count('}'):
            logger.error(f"❌ Mismatched braces: {{ count={json_content.count('{')}, }} count={json_content.count('}')}")
        if json_content.count('[') != json_content.count(']'):
            logger.error(f"❌ Mismatched brackets: [ count={json_content.count('[')}, ] count={json_content.count(']')}")
        
        try:
            response_json = json.loads(json_content)
            logger.info("✅ Successfully parsed JSON")
            logger.info(f"✅ Parsed JSON keys: {list(response_json.keys()) if isinstance(response_json, dict) else 'Not a dict'}")
            
            # Check if this is a function calling format and convert it
            if isinstance(response_json, dict) and "name" in response_json and "arguments" in response_json:
                logger.info(f"🔧 Detected function calling format, converting to schema format")
                tool_name = response_json["name"]
                tool_args = response_json["arguments"]
                
                # Convert function calling format to schema format
                converted_json = {
                    "reasoning_steps": ["Using function calling format", f"Preparing to execute {tool_name}"],
                    "current_situation": f"Executing {tool_name}",
                    "plan_status": "Function call in progress",
                    "enough_data": False,
                    "remaining_steps": [f"Execute {tool_name}"],
                    "task_completed": False,
                    "function": {
                        "tool_name_discriminator": tool_name,
                        **tool_args
                    }
                }
                logger.info(f"🔧 Converted function calling format to schema format")
                response_json = converted_json
            
            # Store the original JSON for later use
            self._original_response_json = response_json
            # Create NextStepToolStub instance from JSON
            reasoning = NextStepToolStub.model_validate(response_json)
            logger.info("✅ Successfully created NextStepToolStub instance")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"❌ Failed to parse JSON: {e}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            logger.error(f"❌ Error details: {str(e)}")
            logger.error(f"❌ Raw content that failed to parse: '{json_content}'")
            logger.error(f"❌ Original content: '{response_content}'")
            logger.error(f"❌ Content before markdown cleanup: '{original_json_content}'")
            
            # Try to identify the problematic character
            if hasattr(e, 'pos') and e.pos is not None:
                logger.error(f"❌ Error position: {e.pos}")
                if e.pos < len(json_content):
                    start = max(0, e.pos - 50)
                    end = min(len(json_content), e.pos + 50)
                    logger.error(f"❌ Content around error position: '{json_content[start:end]}'")
                    logger.error(f"❌ Character at error position: '{json_content[e.pos] if e.pos < len(json_content) else 'EOF'}'")
            
            raise ValueError(f"Failed to parse model response as JSON: {e}")
        # we are not fully sure if it should be in conversation or not. Looks like not necessary data
        # self.conversation.append({"role": "assistant", "content": reasoning.model_dump_json(exclude={"function"})})
        self._log_reasoning(reasoning)
        return reasoning

    def _create_tool_data_with_required_fields(self, tool_name: str, reasoning: NextStepToolStub) -> dict:
        """Create tool data with required fields based on tool type."""
        base_data = {"tool_name_discriminator": tool_name}
        
        if tool_name == "websearchtool":
            # Generate search query from task and remaining steps
            search_query = self.task
            if reasoning.remaining_steps:
                # Use the first remaining step as search query
                search_query = reasoning.remaining_steps[0]
            
            base_data.update({
                "reasoning": f"Searching for information about: {search_query}",
                "query": search_query,
                "max_results": 10,
                "scrape_content": True
            })
        elif tool_name == "createreporttool":
            base_data.update({
                "reasoning": "Creating final report with research findings",
                "title": f"Research Report: {self.task[:50]}...",
                "user_request_language_reference": self.task,
                "content": "Research completed successfully",
                "confidence": "high"
            })
        elif tool_name == "clarificationtool":
            base_data.update({
                "reasoning": "Need clarification to better understand the request",
                "unclear_terms": [self.task],
                "assumptions": ["The request might need more specific details"],
                "questions": [f"Could you provide more details about '{self.task}'?"]
            })
        elif tool_name == "generateplantool":
            base_data.update({
                "reasoning": "Creating research plan for the task",
                "research_goal": self.task,
                "planned_steps": reasoning.remaining_steps or ["Research the topic", "Analyze findings", "Create report"],
                "search_strategies": ["Web search", "Source analysis"]
            })
        elif tool_name == "finalreporttool":
            base_data.update({
                "reasoning": "Presenting final report to user",
                "report_filepath": "",  # Will be set by the system
                "report_title": f"Final Report: {self.task[:50]}...",
                "report_summary": "Research completed successfully"
            })
        
        return base_data

    async def _select_action_phase(self, reasoning: NextStepToolStub) -> BaseTool:
        function_data = reasoning.function
        logger.info(f"🔧 Function data type: {type(function_data)}")
        logger.info(f"🔧 Function data: {function_data}")
        logger.info(f"🔧 Reasoning object: {reasoning}")
        logger.info(f"🔧 Available tools: {[tool.tool_name for tool in self.toolkit if hasattr(tool, 'tool_name')]}")
        
        # Initialize tool_name with a default value
        tool_name = None
        
        # Debug: Log detailed information about function_data
        logger.info(f"🔍 DEBUG function_data: type={type(function_data)}")
        if hasattr(function_data, '__dict__'):
            logger.info(f"🔍 DEBUG function_data attributes: {function_data.__dict__}")
        if isinstance(function_data, dict):
            logger.info(f"🔍 DEBUG function_data dict keys: {list(function_data.keys())}")
        
        if isinstance(function_data, dict):
            # Convert dict to actual tool instance
            tool_name = function_data.get("tool_name_discriminator")
            logger.info(f"🔍 DEBUG: Found tool_name from dict: {tool_name}")
            if not tool_name:
                # Handle empty function dict - this means LLM didn't provide proper function data
                logger.warning("⚠️ Empty function data from LLM - this indicates a parsing or prompt issue")
                logger.warning(f"⚠️ function_data content: {function_data}")
                # Try to determine tool based on reasoning context
                if reasoning.enough_data and len(self._context.sources) > 0:
                    tool_name = "createreporttool"
                    logger.info(f"🔧 Inferred tool_name from context: {tool_name}")
                elif not reasoning.enough_data and len(reasoning.remaining_steps) > 0:
                    tool_name = "websearchtool"
                    logger.info(f"🔧 Inferred tool_name from context: {tool_name}")
                else:
                    tool_name = "clarificationtool"
                    logger.info(f"🔧 Inferred tool_name from context: {tool_name}")
                
                # Update function_data with inferred tool and required fields
                function_data = self._create_tool_data_with_required_fields(tool_name, reasoning)
                logger.info(f"🔧 Updated function_data dict with inferred tool: {tool_name}")
        elif hasattr(function_data, 'tool_name_discriminator'):
            # Handle case when function_data is already a BaseTool instance with discriminator
            tool_name = function_data.tool_name_discriminator
            logger.info(f"🔍 DEBUG: Found tool_name from discriminator: {tool_name}")
        elif hasattr(function_data, 'tool_name'):
            # Handle case when function_data is a BaseTool instance
            tool_name = function_data.tool_name
            logger.info(f"🔍 DEBUG: Found tool_name from tool_name: {tool_name}")
            if not tool_name:
                # Handle empty BaseTool - this means LLM didn't provide proper function data
                logger.warning("⚠️ Empty BaseTool from LLM - this indicates a parsing or prompt issue")
                logger.warning(f"⚠️ function_data content: {function_data}")
                # Try to determine tool based on reasoning context
                if reasoning.enough_data and len(self._context.sources) > 0:
                    tool_name = "createreporttool"
                    logger.info(f"🔧 Inferred tool_name from context: {tool_name}")
                elif not reasoning.enough_data and len(reasoning.remaining_steps) > 0:
                    tool_name = "websearchtool"
                    logger.info(f"🔧 Inferred tool_name from context: {tool_name}")
                else:
                    tool_name = "clarificationtool"
                    logger.info(f"🔧 Inferred tool_name from context: {tool_name}")
                
                # Cannot modify ClassVar tool_name on BaseTool instance
                # Instead, we'll create a new dict with the inferred tool and required fields
                logger.info(f"🔧 Converting BaseTool to dict with inferred tool: {tool_name}")
                function_data = self._create_tool_data_with_required_fields(tool_name, reasoning)
        else:
            # Handle unexpected types
            logger.error(f"❌ Unexpected function_data type: {type(function_data)}")
            logger.error(f"❌ function_data value: {function_data}")
            raise ValueError(f"Expected function_data to be dict or BaseTool, got {type(function_data)}")
        
        logger.info(f"🔍 DEBUG: Final tool_name: {tool_name}")
        
        # Handle case when tool_name is still None
        if tool_name is None:
            logger.error(f"❌ tool_name is None after all checks")
            logger.error(f"❌ function_data type: {type(function_data)}")
            logger.error(f"❌ function_data value: {function_data}")
            raise ValueError("Unable to determine tool_name from function_data")
            
        # CRITICAL: Prevent agentcompletiontool when we have sources and enough data
        logger.info(f"🔍 BLOCKING CHECK: tool_name={tool_name}, sources_count={len(self._context.sources)}, enough_data={reasoning.enough_data}")
        logger.info(f"🔍 CONTEXT SOURCES: {[s.url if hasattr(s, 'url') else str(s) for s in self._context.sources]}")
        
        # Check if we have a report filepath and should present final report
        if hasattr(self._context, 'report_filepath') and self._context.report_filepath:
            logger.info("📋 Report filepath exists - forcing FinalReportTool")
            tool_name = "finalreporttool"
            # Update function_data based on its type
            if isinstance(function_data, dict):
                function_data["tool_name_discriminator"] = "finalreporttool"
                # Add required fields if not present
                if "reasoning" not in function_data:
                    function_data["reasoning"] = "Presenting final report to user"
                if "report_filepath" not in function_data:
                    function_data["report_filepath"] = self._context.report_filepath
                if "report_title" not in function_data:
                    function_data["report_title"] = f"Final Report: {self.task[:50]}..."
                if "report_summary" not in function_data:
                    function_data["report_summary"] = "Research completed successfully"
            else:
                # Convert BaseTool to dict to avoid ClassVar issues
                function_data = {
                    "tool_name_discriminator": "finalreporttool",
                    "reasoning": "Presenting final report to user",
                    "report_filepath": self._context.report_filepath,
                    "report_title": f"Final Report: {self.task[:50]}...",
                    "report_summary": "Research completed successfully"
                }
        elif (tool_name == "agentcompletiontool" and 
              len(self._context.sources) > 0 and 
              reasoning.enough_data):
            logger.info("🚫 BLOCKING agentcompletiontool - we have sources and enough data, must create report")
            tool_name = "createreporttool"
            # Update function_data based on its type
            if isinstance(function_data, dict):
                function_data["tool_name_discriminator"] = "createreporttool"
            else:
                # Convert BaseTool to dict to avoid ClassVar issues
                function_data = {"tool_name_discriminator": "createreporttool"}
        else:
            logger.info(f"🔍 BLOCKING SKIPPED: tool_name={tool_name}, sources_count={len(self._context.sources)}, enough_data={reasoning.enough_data}")
            
            # Find the tool class by name
            tool_class = None
            for tool in self.toolkit:
                if hasattr(tool, 'tool_name') and tool.tool_name == tool_name:
                    tool_class = tool
                    break
            
            if not tool_class:
                raise ValueError(f"Tool class not found for name: {tool_name}")
            
            # If we're creating a report (either forced or originally chosen), prepare the data
            if tool_name == "createreporttool":
                logger.info("🔧 Creating report with research findings")
                # Get the answer content from the original function data
                if isinstance(function_data, dict):
                    answer_content = function_data.get("final_answer", function_data.get("answer", function_data.get("response", "Research completed")))
                else:
                    # Handle BaseTool object
                    answer_content = getattr(function_data, "answer", getattr(function_data, "final_answer", getattr(function_data, "response", "Research completed")))
                # Create report data from the completion data
                function_data = {
                    "tool_name_discriminator": "createreporttool",
                    "reasoning": "Creating final report with research findings",
                    "title": f"Research Report: {self.task[:50]}...",
                    "user_request_language_reference": self.task,
                    "content": answer_content,
                    "confidence": "high",
                    "sources_count": len(self._context.sources),
                    "word_count": len(answer_content),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                logger.info(f"🔧 Report creation data: {function_data}")
            
            # Create tool instance from the data
            try:
                logger.info(f"Creating tool instance for {tool_name} with data: {function_data}")
                tool = tool_class.model_validate(function_data)
                logger.info(f"Created tool instance: {type(tool)} - {tool}")
            except Exception as e:
                logger.error(f"Failed to create tool instance: {e}")
                raise ValueError(f"Failed to create tool instance: {e}")
            
            try:
                # Use the original JSON response that we saved in _reasoning_phase
                if not hasattr(self, '_original_response_json'):
                    logger.error("❌ No original response JSON found")
                    raise ValueError("No original response JSON found")
                
                logger.info(f"🔧 Using original response JSON: {self._original_response_json}")
                function_data_dict = self._original_response_json.get("function", {})
                logger.info(f"🔧 Function data dict: {function_data_dict}")
                tool_name = function_data_dict.get("tool_name_discriminator")
                logger.info(f"🔧 Tool name from original JSON: {tool_name}")
                
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
                        
                        # Map different query field names to standard 'query' field
                        if "search_query" in function_data_dict:
                            function_data_dict["query"] = function_data_dict["search_query"]
                            del function_data_dict["search_query"]
                        elif "queries" in function_data_dict and isinstance(function_data_dict["queries"], list):
                            # Use the first query as the main query
                            function_data_dict["query"] = function_data_dict["queries"][0]
                            del function_data_dict["queries"]
                        elif "search_queries" in function_data_dict and isinstance(function_data_dict["search_queries"], list):
                            # Use the first query as the main query
                            function_data_dict["query"] = function_data_dict["search_queries"][0]
                            del function_data_dict["search_queries"]
                        
                        # Add missing required fields
                        if "query" not in function_data_dict:
                            function_data_dict["query"] = "general information search"
                        
                        if "reasoning" not in function_data_dict:
                            function_data_dict["reasoning"] = f"Searching for {function_data_dict.get('query', 'information')}"
                        
                        if "plan_adapted" not in function_data_dict:
                            function_data_dict["plan_adapted"] = False
                        
                        if "scrape_content" not in function_data_dict:
                            function_data_dict["scrape_content"] = False
                    
                    elif tool_name == "clarificationtool":
                        logger.info(f"🔧 Processing ClarificationTool with data: {function_data_dict}")
                        
                        # Handle question field (single) and convert to questions (plural)
                        if "question" in function_data_dict:
                            # Convert single question to questions list
                            function_data_dict["questions"] = [function_data_dict["question"]]
                            del function_data_dict["question"]
                            logger.info(f"🔧 Converted single question to questions list")
                        
                        # Handle query field (single) and convert to questions (plural)
                        if "query" in function_data_dict:
                            # Convert single query to questions list
                            function_data_dict["questions"] = [function_data_dict["query"]]
                            del function_data_dict["query"]
                            logger.info(f"🔧 Converted single query to questions list")
                        
                        # Handle message field (single) and convert to questions (plural)
                        if "message" in function_data_dict:
                            # Convert single message to questions list
                            function_data_dict["questions"] = [function_data_dict["message"]]
                            del function_data_dict["message"]
                            logger.info(f"🔧 Converted single message to questions list")
                        
                        # Handle questions field for ClarificationTool
                        if "questions" in function_data_dict and isinstance(function_data_dict["questions"], list):
                            # Keep the questions as they are - don't convert to unclear_terms
                            logger.info(f"🔧 Questions field already exists: {function_data_dict['questions']}")
                            pass
                        
                        # Add missing required fields
                        if "reasoning" not in function_data_dict:
                            function_data_dict["reasoning"] = "Need clarification on user requirements"
                        
                        if "unclear_terms" not in function_data_dict:
                            function_data_dict["unclear_terms"] = ["User requirements"]
                        
                        if "assumptions" not in function_data_dict:
                            function_data_dict["assumptions"] = ["User may need more specific information", "Additional context required"]
                        
                        # Generate specific questions based on user request if no questions provided
                        if "questions" not in function_data_dict or not function_data_dict["questions"]:
                            # Generate specific questions based on the user's request
                            user_request = self.task
                            specific_questions = self._generate_clarification_questions(user_request)
                            function_data_dict["questions"] = specific_questions
                            logger.info(f"🔧 Generated specific questions based on user request: {specific_questions}")
                        else:
                            # If we have questions but they are generic, try to improve them
                            questions = function_data_dict["questions"]
                            if len(questions) == 1 and any(generic in questions[0].lower() for generic in ["could you provide", "what specific", "any particular"]):
                                # These are generic questions, replace with specific ones
                                user_request = self.task
                                specific_questions = self._generate_clarification_questions(user_request)
                                function_data_dict["questions"] = specific_questions
                                logger.info(f"🔧 Replaced generic questions with specific ones: {specific_questions}")
                        
                        logger.info(f"🔧 Final ClarificationTool data: {function_data_dict}")
                    
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
                        logger.info(f"🔧 Processing AgentCompletionTool with data: {function_data_dict}")
                        
                        # Handle missing required fields for AgentCompletionTool
                        if "reasoning" not in function_data_dict:
                            function_data_dict["reasoning"] = "Task completion reasoning"
                        
                        if "completed_steps" not in function_data_dict:
                            function_data_dict["completed_steps"] = ["Research completed"]
                        
                        if "status" not in function_data_dict:
                            function_data_dict["status"] = "completed"
                        
                        # Handle response field (from agent) and convert to answer field
                        if "response" in function_data_dict:
                            function_data_dict["answer"] = function_data_dict["response"]
                            del function_data_dict["response"]
                            logger.info(f"🔧 Converted response field to answer field")
                        
                        # Handle final_answer field (from agent) and convert to answer field
                        if "final_answer" in function_data_dict:
                            function_data_dict["answer"] = function_data_dict["final_answer"]
                            del function_data_dict["final_answer"]
                            logger.info(f"🔧 Converted final_answer field to answer field")
                        
                        # Handle answer and sources fields
                        if "answer" not in function_data_dict:
                            function_data_dict["answer"] = "Research completed successfully"
                        
                        if "sources" not in function_data_dict:
                            function_data_dict["sources"] = "Sources available in research context"
                        elif isinstance(function_data_dict["sources"], list):
                            # Convert list to string
                            logger.info(f"🔧 Converting sources list to string: {function_data_dict['sources']}")
                            function_data_dict["sources"] = ", ".join(function_data_dict["sources"])
                            logger.info(f"🔧 Converted sources to: {function_data_dict['sources']}")
                        
                        logger.info(f"🔧 Final AgentCompletionTool data: {function_data_dict}")
                    
                    elif tool_name == "createreporttool":
                        # Handle missing required fields for CreateReportTool
                        if "reasoning" not in function_data_dict:
                            function_data_dict["reasoning"] = "Creating comprehensive research report"
                        
                        if "title" not in function_data_dict:
                            function_data_dict["title"] = "Research Report"
                        
                        if "user_request_language_reference" not in function_data_dict:
                            function_data_dict["user_request_language_reference"] = self.task
                        
                        if "content" not in function_data_dict:
                            # Generate comprehensive report content from research data
                            report_content = self._generate_report_content()
                            function_data_dict["content"] = report_content
                        
                        if "confidence" not in function_data_dict:
                            function_data_dict["confidence"] = "high"
                    
                    elif tool_name == "finalreporttool":
                        # Handle missing required fields for FinalReportTool
                        if "reasoning" not in function_data_dict:
                            function_data_dict["reasoning"] = "Presenting final report to user"
                        
                        if "report_filepath" not in function_data_dict and hasattr(self._context, 'report_filepath'):
                            function_data_dict["report_filepath"] = self._context.report_filepath
                        
                        if "report_title" not in function_data_dict and hasattr(self._context, 'report_title'):
                            function_data_dict["report_title"] = self._context.report_title
                        
                        if "report_summary" not in function_data_dict:
                            function_data_dict["report_summary"] = f"Research completed with {len(self._context.sources)} sources and {len(self._context.searches)} searches"
                        
                        # Map confidence values to valid enum values
                        if "confidence" in function_data_dict:
                            confidence_value = function_data_dict["confidence"].lower()
                            if "высокая" in confidence_value or "high" in confidence_value:
                                function_data_dict["confidence"] = "high"
                            elif "средняя" in confidence_value or "medium" in confidence_value:
                                function_data_dict["confidence"] = "medium"
                            elif "низкая" in confidence_value or "low" in confidence_value:
                                function_data_dict["confidence"] = "low"
                            else:
                                function_data_dict["confidence"] = "medium"  # default fallback
                        else:
                            function_data_dict["confidence"] = "medium"
                        
                        if "user_request_language_reference" not in function_data_dict:
                            function_data_dict["user_request_language_reference"] = "English"
                        
                        # Remove fields that are not in the schema
                        if "sources" in function_data_dict:
                            del function_data_dict["sources"]
                        if "analysis" in function_data_dict:
                            del function_data_dict["analysis"]
                    
                    elif tool_name == "reasoningtool":
                        # Handle missing required fields for ReasoningTool
                        if "reasoning_steps" not in function_data_dict:
                            function_data_dict["reasoning_steps"] = ["Step 1", "Step 2", "Step 3"]
                        else:
                            # Ensure reasoning_steps doesn't exceed max length of 6
                            if len(function_data_dict["reasoning_steps"]) > 6:
                                function_data_dict["reasoning_steps"] = function_data_dict["reasoning_steps"][:6]
                        
                        if "current_situation" not in function_data_dict:
                            function_data_dict["current_situation"] = "Current research situation"
                        
                        if "plan_status" not in function_data_dict:
                            function_data_dict["plan_status"] = "Plan status"
                        
                        if "remaining_steps" not in function_data_dict:
                            function_data_dict["remaining_steps"] = []
                        else:
                            # Ensure remaining_steps doesn't exceed max length of 3
                            if len(function_data_dict["remaining_steps"]) > 3:
                                function_data_dict["remaining_steps"] = function_data_dict["remaining_steps"][:3]
                        
                        if "task_completed" not in function_data_dict:
                            function_data_dict["task_completed"] = False
                    
                    # Apply length constraints to all list fields that might exceed limits
                    for field_name, max_length in [
                        ("reasoning_steps", 6),
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
                    
                    # Add to conversation and streaming generator
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
                    
                except Exception as e:
                    logger.error(f"Failed to create tool instance: {e}")
                    raise ValueError(f"Failed to create tool instance: {e}")
                    
            except Exception as e:
                logger.error(f"Failed to extract tool data from original JSON: {e}")
                raise ValueError(f"Failed to extract tool data from original JSON: {e}")

    async def _action_phase(self, tool: BaseTool) -> str:
        result = tool(self._context)
        self.conversation.append(
            {"role": "tool", "content": result, "tool_call_id": f"{self._context.iteration}-action"}
        )
        self.streaming_generator.add_chunk(f"{result}\n")
        self._log_tool_execution(tool, result)
        
        # If we just created a report, prepare for final presentation
        if tool.tool_name == "createreporttool":
            logger.info("🎯 Report created - preparing for final presentation")
            # Store the report filepath for final presentation
            try:
                result_data = json.loads(result)
                if isinstance(result_data, dict) and "filepath" in result_data:
                    self._context.report_filepath = result_data["filepath"]
                    self._context.report_title = result_data.get("title", "Research Report")
                    logger.info(f"📁 Stored report filepath: {self._context.report_filepath}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"⚠️ Could not extract report filepath: {e}")
            
            # Don't mark as completed yet - we need to present the final report
            # The agent will continue to use FinalReportTool
        elif tool.tool_name == "finalreporttool":
            logger.info("🎯 Final report presented - marking task as completed")
            self._context.state = AgentStatesEnum.COMPLETED
        
        return result

    def _generate_report_content(self) -> str:
        """Generate comprehensive report content from research data."""
        content_parts = []
        
        # Add executive summary
        content_parts.append("## Executive Summary")
        content_parts.append(f"This report provides a comprehensive analysis of: {self.task}")
        
        # Add research findings from searches
        if self._context.searches:
            content_parts.append("\n## Research Findings")
            for i, search in enumerate(self._context.searches, 1):
                content_parts.append(f"\n### Finding {i}: {search.query}")
                if search.answer:
                    content_parts.append(search.answer)
                else:
                    content_parts.append("Research conducted on this topic.")
        
        # Add sources analysis
        if self._context.sources:
            content_parts.append("\n## Sources Analysis")
            content_parts.append(f"Total sources analyzed: {len(self._context.sources)}")
            for source in self._context.sources.values():
                content_parts.append(f"- **{source.title}**: {source.snippet[:200]}...")
        
        # Add conclusions
        content_parts.append("\n## Conclusions")
        content_parts.append("Based on the research conducted, the following conclusions can be drawn:")
        content_parts.append("- Research has been completed successfully")
        content_parts.append(f"- {len(self._context.sources)} sources were analyzed")
        content_parts.append(f"- {len(self._context.searches)} search queries were executed")
        
        return "\n".join(content_parts)

    def _generate_clarification_questions(self, user_request: str) -> list[str]:
        """Generate specific clarification questions based on the user's request."""
        # Detect language from user request
        is_russian = any(ord(char) > 127 for char in user_request)
        
        if is_russian:
            # Russian questions
            if "прогулк" in user_request.lower() or "ходьб" in user_request.lower():
                return [
                    "Какой у вас возраст?",
                    "Есть ли у вас какие-либо проблемы со здоровьем?",
                    "Какова ваша цель: поддержание здоровья, похудение или что-то другое?",
                    "Какой у вас текущий уровень физической активности?"
                ]
            elif "диет" in user_request.lower() or "питани" in user_request.lower():
                return [
                    "Есть ли у вас аллергии или непереносимость продуктов?",
                    "Какой у вас образ жизни (сидячий, активный)?",
                    "Есть ли у вас хронические заболевания?",
                    "Какова ваша цель: похудение, набор веса или поддержание?"
                ]
            elif "тренировк" in user_request.lower() or "спорт" in user_request.lower():
                return [
                    "Какой у вас уровень подготовки (новичок, средний, продвинутый)?",
                    "Есть ли у вас травмы или ограничения?",
                    "Какова ваша цель тренировок?",
                    "Сколько времени вы можете уделять тренировкам в неделю?"
                ]
            else:
                return [
                    "Можете ли вы уточнить, что именно вас интересует?",
                    "Есть ли какие-то конкретные аспекты, на которых стоит сосредоточиться?",
                    "Какой у вас уровень знаний в этой области?",
                    "Есть ли какие-то ограничения или предпочтения?"
                ]
        else:
            # English questions
            if "walk" in user_request.lower() or "walking" in user_request.lower():
                return [
                    "What is your age?",
                    "Do you have any health conditions?",
                    "What is your goal: health maintenance, weight loss, or something else?",
                    "What is your current activity level?"
                ]
            elif "diet" in user_request.lower() or "nutrition" in user_request.lower():
                return [
                    "Do you have any food allergies or intolerances?",
                    "What is your lifestyle (sedentary, active)?",
                    "Do you have any chronic conditions?",
                    "What is your goal: weight loss, weight gain, or maintenance?"
                ]
            elif "workout" in user_request.lower() or "exercise" in user_request.lower():
                return [
                    "What is your fitness level (beginner, intermediate, advanced)?",
                    "Do you have any injuries or limitations?",
                    "What is your workout goal?",
                    "How much time can you dedicate to workouts per week?"
                ]
            else:
                return [
                    "Could you clarify what specifically interests you?",
                    "Are there any particular aspects you'd like to focus on?",
                    "What is your knowledge level in this area?",
                    "Are there any constraints or preferences?"
                ]
