import asyncio
from mcp import StdioServerParameters
from converse_agent import ConverseAgent
from converse_tools import ConverseToolManager
from mcp_client import MCPClient

async def main():
    """
    Main function that sets up and runs an interactive AI agent with tool integration.
    The agent can process user prompts and utilize registered tools to perform tasks.
    """
    # Initialize model configuration
    model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    # model_id = "mistral.mistral-large-2407-v1:0"
    # model_id = "us.meta.llama3-2-90b-instruct-v1:0"
    
    # Set up the agent and tool manager
    agent = ConverseAgent(model_id)
    agent.tools = ConverseToolManager()

    # Define the agent's behavior through system prompt
    agent.system_prompt = """You are a helpful assistant that can use tools to help you answer 
questions and perform tasks."""

    # Create server parameters for SQLite configuration
    server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-sqlite", "--db-path", "~/test.db"],
        env=None
    )

    # Initialize MCP client with server parameters
    async with MCPClient(server_params) as mcp_client:

        # Fetch available tools from the MCP client
        tools_response = await mcp_client.get_available_tools()
        print(f"Available tools response: {tools_response}")  # Debug log
        
        # Extract the tools list from the response
        tools = []
        if hasattr(tools_response, 'tools'):
            tools = tools_response.tools
        elif isinstance(tools_response, dict) and 'tools' in tools_response:
            tools = tools_response['tools']
        else:
            print(f"Warning: Unexpected tools response format: {type(tools_response)}")
            print(f"Tools response content: {tools_response}")

        if not tools:
            print("Warning: No tools found in response")
            # Register at least one default tool to satisfy Bedrock's requirement
            agent.tools.register_tool(
                name="help",
                func=lambda x: "This is a help function",
                description="Get help about available tools",
                input_schema={'json': {
                    "type": "object",
                    "properties": {},
                    "required": []
                }}
            )
            print("Registered default help tool")

        # Register each available tool with the agent
        for tool in tools:
            try:
                # Tools should be Tool objects with name, description, and inputSchema attributes
                name = getattr(tool, 'name', None)
                description = getattr(tool, 'description', "Tool description not available")
                schema = getattr(tool, 'inputSchema', None)

                if name is None:
                    print(f"Skipping tool without name: {tool}")
                    continue

                # Ensure input schema follows Bedrock Converse API requirements
                input_schema = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }

                if schema:
                    if isinstance(schema, dict):
                        input_schema["properties"] = schema.get("properties", {})
                        input_schema["required"] = schema.get("required", [])
                
                agent.tools.register_tool(
                    name=name,
                    func=mcp_client.call_tool,
                    description=description,
                    input_schema={'json': input_schema}
                )
                print(f"Registered tool: {name} with schema: {input_schema}")  # Debug log
            except Exception as e:
                print(f"Error registering tool: {e}")  # Debug log
                continue

        # Start interactive prompt loop
        while True:
            try:
                # Get user input and check for exit commands
                user_prompt = input("\nEnter your prompt (or 'quit' to exit): ")
                if user_prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                # Process the prompt and display the response
                response = await agent.invoke_with_prompt(user_prompt)
                print("\nResponse:", response)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError occurred: {e}")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 
