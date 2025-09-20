import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8010/v1", api_key="dummy")

# Step 1: Initial research request
print("Starting research...")
response = client.chat.completions.create(
    model="sgr-agent",
    messages=[{"role": "user", "content": "Research AI market trends"}],
    stream=True,
    temperature=0,
)

agent_id = None
clarification_questions = []
tool_calls_data = []

# Process streaming response
for chunk in response:
    # Extract agent ID from model field
    if chunk.model and chunk.model.startswith("sgr_agent_"):
        agent_id = chunk.model
        print(f"\nAgent ID: {agent_id}")

    # Check for tool calls
    if chunk.choices[0].delta.tool_calls:
        for tool_call in chunk.choices[0].delta.tool_calls:
            if tool_call.function:
                print(f"\n\n🔧 Tool call: {tool_call.function.name}")
                if tool_call.function.arguments:
                    try:
                        args = json.loads(tool_call.function.arguments)
                        print(f"📋 Arguments: {json.dumps(args, indent=2, ensure_ascii=False)}")
                        
                        # Special handling for different tools
                        if tool_call.function.name == "clarificationtool":
                            clarification_questions = args.get("questions", [])
                            print(f"\n❓ Clarification needed:")
                            for i, question in enumerate(clarification_questions, 1):
                                print(f"   {i}. {question}")
                        elif tool_call.function.name == "create_report":
                            print(f"\n📝 REPORT CREATED:")
                            print(f"   Title: {args.get('title', 'N/A')}")
                            print(f"   Confidence: {args.get('confidence', 'N/A')}")
                            if 'content' in args:
                                content_preview = args['content'][:200] + "..." if len(args['content']) > 200 else args['content']
                                print(f"   Content Preview: {content_preview}")
                        elif tool_call.function.name == "web_search":
                            print(f"\n🔍 WEB SEARCH:")
                            print(f"   Query: {args.get('query', 'N/A')}")
                            print(f"   Max Results: {args.get('max_results', 'N/A')}")
                            print(f"   Scrape Content: {args.get('scrape_content', 'N/A')}")
                            
                    except json.JSONDecodeError as e:
                        print(f"❌ Error parsing tool call arguments: {e}")
                        print(f"Raw arguments: {tool_call.function.arguments}")

    # Print content
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# Step 2: Handle clarification if needed
if clarification_questions and agent_id:
    # Provide clarification
    clarification = "Focus on LLM market trends for 2024-2025, global perspective"
    print(f"\nProviding clarification: {clarification}")

    # Continue with agent ID
    response = client.chat.completions.create(
        model=agent_id,  # Use agent ID as model
        messages=[{"role": "user", "content": clarification}],
        stream=True,
        temperature=0,
    )

    # Print final response
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")
        
        # Check for additional tool calls
        if chunk.choices[0].delta.tool_calls:
            for tool_call in chunk.choices[0].delta.tool_calls:
                if tool_call.function:
                    print(f"\n\nTool call: {tool_call.function.name}")
                    if tool_call.function.arguments:
                        try:
                            args = json.loads(tool_call.function.arguments)
                            print(f"Arguments: {json.dumps(args, indent=2, ensure_ascii=False)}")
                            
                            # Special handling for create_report tool
                            if tool_call.function.name == "create_report":
                                print(f"\n📝 REPORT CREATED:")
                                print(f"Title: {args.get('title', 'N/A')}")
                                print(f"Confidence: {args.get('confidence', 'N/A')}")
                                if 'content' in args:
                                    content_preview = args['content'][:200] + "..." if len(args['content']) > 200 else args['content']
                                    print(f"Content Preview: {content_preview}")
                                
                        except json.JSONDecodeError:
                            print(f"Raw arguments: {tool_call.function.arguments}")

print("\n\nResearch completed!")