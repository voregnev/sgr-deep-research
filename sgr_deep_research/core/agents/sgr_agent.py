import logging
import uuid
from typing import Type

from sgr_deep_research.core.agents.base_agent import BaseAgent
from sgr_deep_research.core.base_tool import BaseTool
from sgr_deep_research.settings import get_config
from sgr_deep_research.tools import (
    AgentCompletionTool,
    ClarificationTool,
    CreateReportTool,
    GeneratePlanTool,
    NextStepToolsBuilder,
    NextStepToolStub,
    ReasoningTool,
    WebSearchTool,
)

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
        max_iterations: int = 10,
        max_searches: int = 4,
    ):
        super().__init__(
            task=task,
            toolkit=toolkit,
            max_clarifications=max_clarifications,
            max_iterations=max_iterations,
        )

        self.id = f"sgr_agent_{uuid.uuid4()}"

        self.toolkit = [*self.tools_registry.get_tools(), *(toolkit or [])]
        self.toolkit.remove(ReasoningTool)  # we use our own reasoning scheme
        self.max_searches = max_searches

    async def _prepare_tools(self) -> Type[NextStepToolStub]:
        """Prepare tool classes with current context limits."""
        tools = set(self.toolkit)
        
        # Remove GeneratePlanTool after first iteration to prevent loops
        if self._context.iteration > 1:
            tools -= {GeneratePlanTool}
            
        if self._context.iteration >= self.max_iterations:
            tools = {
                CreateReportTool,
                AgentCompletionTool,
            }
        if self._context.clarifications_used >= self.max_clarifications:
            tools -= {
                ClarificationTool,
            }
        if self._context.searches_used >= self.max_searches:
            tools -= {
                WebSearchTool,
            }
        return NextStepToolsBuilder.build_NextStepTools(list(tools))

    async def _reasoning_phase(self) -> NextStepToolStub:
        try:
            async with self.openai_client.chat.completions.stream(
                model=config.openai.model,
                response_format=await self._prepare_tools(),
                messages=await self._prepare_context(),
                max_tokens=config.openai.max_tokens,
                temperature=config.openai.temperature,
            ) as stream:
                async for event in stream:
                    if event.type == "chunk":
                        content = event.chunk.choices[0].delta.content
                        self.streaming_generator.add_chunk(content)
            reasoning: NextStepToolStub = (await stream.get_final_completion()).choices[0].message.parsed  # type: ignore
            # we are not fully sure if it should be in conversation or not. Looks like not necessary data
            # self.conversation.append({"role": "assistant", "content": reasoning.model_dump_json(exclude={"function"})})
            self._log_reasoning(reasoning)
            return reasoning
        except Exception as e:
            if "length limit was reached" in str(e).lower():
                logger.warning(f"⚠️ Token limit reached, retrying with reduced context. Error: {str(e)}")
                # Try to reduce context and retry
                return await self._reasoning_phase_with_reduced_context()
            else:
                raise e

    async def _reasoning_phase_with_reduced_context(self) -> NextStepToolStub:
        """Fallback method with reduced context when token limit is reached."""
        logger.info("🔄 Retrying with reduced context and increased token limit")
        
        # Reduce conversation history to last 3 messages
        original_conversation = self.conversation.copy()
        if len(self.conversation) > 3:
            self.conversation = self.conversation[-3:]
            logger.info(f"📝 Reduced conversation from {len(original_conversation)} to {len(self.conversation)} messages")
        
        try:
            async with self.openai_client.chat.completions.stream(
                model=config.openai.model,
                response_format=await self._prepare_tools(),
                messages=await self._prepare_context(),
                max_tokens=config.openai.max_tokens * 2,  # Double the token limit
                temperature=config.openai.temperature,
            ) as stream:
                async for event in stream:
                    if event.type == "chunk":
                        content = event.chunk.choices[0].delta.content
                        self.streaming_generator.add_chunk(content)
            reasoning: NextStepToolStub = (await stream.get_final_completion()).choices[0].message.parsed  # type: ignore
            self._log_reasoning(reasoning)
            return reasoning
        except Exception as e:
            # Restore original conversation if retry fails
            self.conversation = original_conversation
            logger.error(f"❌ Retry with reduced context also failed: {str(e)}")
            raise e

    async def _select_action_phase(self, reasoning: NextStepToolStub) -> BaseTool:
        tool = reasoning.function
        if not isinstance(tool, BaseTool):
            raise ValueError("Selected tool is not a valid BaseTool instance")
        
        # Log tool selection for debugging
        logger.info(f"🔧 Selected tool: {tool.tool_name} (iteration {self._context.iteration})")
        
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
