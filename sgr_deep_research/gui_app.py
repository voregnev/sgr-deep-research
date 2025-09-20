#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGR Research Agent - Chainlit GUI Interface
Beautiful web interface for the research agent using new SGR architecture
"""

import json
import logging
import os
import asyncio
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

import chainlit as cl
from dotenv import load_dotenv

# Setup logging
logger = logging.getLogger(__name__)

# Load environment variables (optional, config.yaml takes priority)
load_dotenv()

# Local modules
from sgr_deep_research.core.agents import SGRResearchAgent
from sgr_deep_research.core.models import ResearchContext, AgentStatesEnum
from sgr_deep_research.core.prompts import PromptLoader
from sgr_deep_research.settings import get_config


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = get_config()


# =============================================================================
# CHAINLIT HELPER FUNCTIONS
# =============================================================================

def create_fresh_context() -> ResearchContext:
    """Create fresh research context."""
    return ResearchContext()


async def display_final_report(result_data):
    """Display final report with full content from file."""
    try:
        report_content = result_data.get("content", "")
        report_title = result_data.get("title", "Research Report")
        report_summary = result_data.get("summary", "")
        
        # Create beautiful report display
        report_display = f"""# 📋 {report_title}

{report_summary}

---

## 📄 Full Report Content

{report_content}

---

**📊 Report Statistics:**
- **Sources analyzed:** {result_data.get('sources_count', 0)}
- **Searches performed:** {result_data.get('searches_count', 0)}
- **Generated:** {result_data.get('timestamp', 'N/A')}
- **File location:** {result_data.get('filepath', 'N/A')}
"""
        
        # Send the report
        report_msg = cl.Message(
            author="report",
            content=report_display,
        )
        await report_msg.send()
        
        # Add copy button for the report
        await cl.Message(
            author="actions",
            content="📋 **Copy Report:**",
            actions=[
                cl.Action(
                    name="copy_report",
                    value="Copy Full Report",
                    payload={"content": report_content}
                )
            ]
        ).send()
        
    except Exception as e:
        logger.error(f"❌ Error displaying final report: {e}")
        await cl.Message(
            author="error",
            content=f"Error displaying final report: {str(e)}"
        ).send()

async def display_agent_status(agent: SGRResearchAgent, context: ResearchContext) -> None:
    """Display agent status beautifully."""

    status_info = f"""
**🎯 Agent Status:** {agent._context.state.value}

**📊 Statistics:**
- Searches done: {len(context.searches)}
- Sources found: {len(context.sources)}
- Clarifications used: {context.clarifications_used}

**📝 Current Task:** {agent.task}
    """

    msg = cl.Message(
        author="agent_status",
        content=status_info,
    )
    await msg.send()




# =============================================================================
# MAIN EXECUTION FUNCTIONS
# =============================================================================

async def run_research_agent(task: str, context: ResearchContext) -> None:
    """Run the SGR research agent with GUI integration."""
    
    # Create agent
    agent = SGRResearchAgent(
        task=task,
        max_searches=CONFIG.execution.max_steps,
        max_iterations=CONFIG.execution.max_steps
    )
    
    # Store active agent in session
    cl.user_session.set("active_agent", agent)
    
    # Display initial status
    await display_agent_status(agent, context)
    
    try:
        # Start agent execution in background
        import asyncio
        agent_task = asyncio.create_task(agent.execute())
        
        # Monitor agent state in real-time
        clarification_sent = False
        while not agent_task.done():
            await asyncio.sleep(0.5)  # Check every 500ms
            
            if agent._context.state == AgentStatesEnum.WAITING_FOR_CLARIFICATION and not clarification_sent:
                # Display the specific questions from ClarificationTool
                questions_content = agent._context.pending_clarification_questions
                print(f"DEBUG: Agent waiting for clarification, questions_content: '{questions_content}'")
                print(f"DEBUG: Questions content length: {len(questions_content)}")
                print(f"DEBUG: Questions content type: {type(questions_content)}")
                
                if questions_content and questions_content.strip():
                    clarification_msg = cl.Message(
                        author="clarification",
                        content=f"The agent needs more information to proceed. Please answer these questions:\n\n{questions_content}"
                    )
                else:
                    clarification_msg = cl.Message(
                        author="clarification",
                        content="The agent needs more information to proceed. Please provide additional details about your request."
                    )
                await clarification_msg.send()
                clarification_sent = True
                print(f"DEBUG: Clarification message sent, breaking loop")
                # Break the loop to wait for user input
                break
        
        # Wait for agent to complete (either after clarification or without needing it)
        await agent_task
        
        # Display the final result
        if agent._context.state == AgentStatesEnum.COMPLETED:
            # Get the final result from the agent's conversation
            final_messages = [msg for msg in agent.conversation if msg.get("role") == "tool"]
            if final_messages:
                final_result = final_messages[-1].get("content", "Research completed successfully")
                
                # Try to parse as JSON to check if it's a report or completion
                try:
                    result_data = json.loads(final_result)
                    logger.info(f"🔍 GUI: Parsed result_data: {result_data}")
                    if isinstance(result_data, dict):
                        # Handle FinalReportTool results
                        if result_data.get("type") == "final_report":
                            await display_final_report(result_data)
                        elif "title" in result_data and "content" in result_data:
                            # This is a report - display it beautifully
                            report_content = f"""# 📝 {result_data.get('title', 'Research Report')}

**📊 Report Details:**
- **Confidence:** {result_data.get('confidence', 'N/A')}
- **Sources:** {result_data.get('sources_count', 0)}
- **Words:** {result_data.get('word_count', 0)}
- **Created:** {result_data.get('timestamp', 'N/A')}

---

## 📄 Report Content

{result_data.get('content', 'No content available')}

---

**💾 Report saved to:** `{result_data.get('filepath', 'N/A')}`
"""
                            
                            # Create message with copy button
                            result_msg = cl.Message(
                                author="research_result",
                                content=report_content
                            )
                            await result_msg.send()
                            
                            # Add copy button for the report content
                            await cl.Message(
                                author="actions",
                                content="📋 **Copy Report as Markdown**",
                                actions=[
                                    cl.Action(
                                        name="copy_report",
                                        payload={"content": result_data.get('content', '')},
                                        label="📋 Copy Report Content"
                                    )
                                ]
                            ).send()
                        elif "answer" in result_data and "sources" in result_data:
                            # This is an AgentCompletionTool result - display the answer
                            answer_content = f"""# ✅ Research Completed Successfully!

## 📋 Research Answer

{result_data.get('answer', 'No answer available')}

---

## 📚 Sources Used

{result_data.get('sources', 'No sources available')}

---

## 📊 Research Summary

**Status:** {result_data.get('status', 'completed')}  
**Completed Steps:** {len(result_data.get('completed_steps', []))}  
**Reasoning:** {result_data.get('reasoning', 'Task completed successfully')}
"""
                            
                            # Create message with copy button
                            result_msg = cl.Message(
                                author="research_result",
                                content=answer_content
                            )
                            await result_msg.send()
                            
                            # Add copy button for the answer content
                            await cl.Message(
                                author="actions",
                                content="📋 **Copy Answer**",
                                actions=[
                                    cl.Action(
                                        name="copy_answer",
                                        payload={"content": result_data.get('answer', '')},
                                        label="📋 Copy Answer"
                                    )
                                ]
                            ).send()
                        else:
                            # Regular JSON result
                            result_msg = cl.Message(
                                author="research_result",
                                content=f"**Research Completed Successfully!**\n\n```json\n{json.dumps(result_data, indent=2, ensure_ascii=False)}\n```"
                            )
                            await result_msg.send()
                    else:
                        # Regular result
                        result_msg = cl.Message(
                            author="research_result",
                            content=f"**Research Completed Successfully!**\n\n{final_result}"
                        )
                        await result_msg.send()
                except json.JSONDecodeError:
                    # Not JSON, display as regular result
                    result_msg = cl.Message(
                        author="research_result",
                        content=f"**Research Completed Successfully!**\n\n{final_result}"
                    )
                    await result_msg.send()
            else:
                result_msg = cl.Message(
                    author="research_result",
                    content="**Research completed successfully!**\n\nCheck the reports directory for detailed results."
                )
                await result_msg.send()
        elif agent._context.state not in [AgentStatesEnum.WAITING_FOR_CLARIFICATION, AgentStatesEnum.COMPLETED]:
            status_msg = cl.Message(
                author="warning",
                content=f"Research ended with status: {agent._context.state.value}"
            )
            await status_msg.send()
                
        # Display final status
        await display_agent_status(agent, context)
        
        # Final statistics
        final_stats = f"""
**📊 Session completed:**

🔎 Searches: {len(context.searches)} | 📚 Sources: {len(context.sources)}
📁 Reports saved to: `./{CONFIG.execution.reports_dir}/`
        """
        await cl.Message(author="statistics", content=final_stats).send()
        
    except Exception as e:
        error_msg = cl.Message(
            author="error",
            content=f"Error during research: {str(e)}"
        )
        await error_msg.send()
    finally:
        # Clear active agent when done
        cl.user_session.set("active_agent", None)


































# =============================================================================
# CHAINLIT EVENT HANDLERS
# =============================================================================


@cl.on_chat_start
async def start_chat():
    """Chat initialization."""

    # Create fresh context
    context = create_fresh_context()

    # Save to session
    cl.user_session.set("context", context)

    # Welcome message
    welcome_msg = f"""
# 🧠 SGR Research Agent - Web Interface

Welcome to the intelligent research system!

**🔍 Agent Capabilities:**
- **Situation Analysis** - smart research planning
- **Web Search** - finding credible information via Tavily
- **Clarification** - questions for better task understanding
- **Report Creation** - detailed research reports with citations
- **File Operations** - read, create, and update local files
- **Directory Management** - browse and organize file structures
- **Quick Answers** - fast responses without full reports
- **Date/Time Awareness** - current date and time information

**🚀 Configuration:**
- Model: `{str(CONFIG.openai.model)}`
- Max Steps: {str(CONFIG.execution.max_steps)}

**Simply ask your research question or request any file operation, and I'll conduct deep analysis!**
    """

    await cl.Message(author="system", content=welcome_msg).send()


@cl.action_callback("copy_report")
async def on_copy_report(action: cl.Action):
    """Handle copy report action."""
    await cl.Message(
        author="system",
        content=f"📋 **Report content copied to clipboard!**\n\n*You can now paste it anywhere you need.*"
    ).send()


@cl.action_callback("copy_answer")
async def on_copy_answer(action: cl.Action):
    """Handle copy answer action."""
    await cl.Message(
        author="system",
        content=f"📋 **Answer content copied to clipboard!**\n\n*You can now paste it anywhere you need.*"
    ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    """Handle user message."""

    # Get context
    context = cl.user_session.get("context", create_fresh_context())
    
    # Check if there's an active agent waiting for clarification
    active_agent = cl.user_session.get("active_agent", None)
    
    user_content = message.content.strip()
    
    # Debug logging
    print(f"DEBUG: Received message: '{user_content}'")
    print(f"DEBUG: Active agent exists: {active_agent is not None}")
    if active_agent:
        print(f"DEBUG: Agent state: {active_agent._context.state}")
        print(f"DEBUG: Agent waiting for clarification: {active_agent._context.state == AgentStatesEnum.WAITING_FOR_CLARIFICATION}")

    if not user_content:
        await cl.Message(
            author="warning",
            content="Please enter your research request.",
        ).send()
        return
    
    # If there's an active agent waiting for clarification, provide it
    if active_agent and active_agent._context.state == AgentStatesEnum.WAITING_FOR_CLARIFICATION:
        try:
            # Log the clarification attempt
            await cl.Message(
                author="process",
                content=f"Providing clarification: {user_content[:100]}..."
            ).send()
            
            await active_agent.provide_clarification(user_content)
            
            # Log successful clarification
            await cl.Message(
                author="success",
                content="Clarification received! Continuing research..."
            ).send()
            
            # Clear the active agent to prevent further clarification requests
            cl.user_session.set("active_agent", None)
            
            # Agent will continue automatically in the background task
            return
        except Exception as e:
            await cl.Message(
                author="error",
                content=f"Error providing clarification: {str(e)}"
            ).send()
            return

    # Check commands
    if user_content.lower() in ["help", "помощь"]:
        help_msg = """
**🆘 Command Help:**

- Simply ask a research question or request file operations
- Agent will automatically conduct analysis
- Supported commands: `help`, `stats`, `clear`

**Example queries:**
- "Research the electric vehicle market in 2024"
- "Analyze latest achievements in AI"
- "Read the contents of report.txt"
- "Create a summary file of our discussion"
- "List files in the current directory"
        """
        await cl.Message(author="help", content=help_msg).send()
        return

    if user_content.lower() in ["stats", "статистика"]:
        stats_msg = f"""
**📊 Session Statistics:**

🔍 **Searches:** {len(context.searches)}
📚 **Sources:** {len(context.sources)}
❓ **Clarification used:** {'Yes' if context.clarification_received else 'No'}
📄 **Reports created:** {len([s for s in context.searches if s.get('report_created')])}
        """
        await cl.Message(author="statistics", content=stats_msg).send()
        return

    if user_content.lower() in ["clear", "очистить"]:
        # Reset context and clear active agent
        context = create_fresh_context()
        cl.user_session.set("context", context)
        cl.user_session.set("active_agent", None)

        await cl.Message(
            author="process",
            content="Context cleared. You can start a new research!",
        ).send()
        return

    # Main research logic
    try:
        # Show progress
        progress_msg = cl.Message(
            author="process",
            content="**Starting research analysis...**",
        )
        await progress_msg.send()

        # Run research agent in background task
        import asyncio
        asyncio.create_task(run_research_agent(user_content, context))

    except Exception as e:
        error_msg = cl.Message(
            author="error",
            content=f"Critical error occurred: {str(e)}",
        )
        await error_msg.send()

    finally:
        # Save updated context
        cl.user_session.set("context", context)


# =============================================================================
# CHAINLIT SETTINGS
# =============================================================================


@cl.set_chat_profiles
async def chat_profile(user=None):
    return [
        cl.ChatProfile(
            name="research",
            markdown_description="🧠 **Research Mode** - full analysis with search and reports",
        ),
    ]


if __name__ == "__main__":
    # Launch Chainlit application
    cl.run()
