# Manus-Inspired Agent with LangGraph and Claude 3.7

This project implements an AI agent architecture inspired by the Manus system, using LangGraph for workflow management and Claude 3.7 as the foundational language model. The agent follows an iterative loop pattern for task execution, with tool selection and execution capabilities.

## Features

- **Agent Loop Architecture**: Implements the Manus agent loop pattern (analyze, select tools, execute, iterate)
- **Tool Integration**: Includes tools for messaging, file operations, shell commands, web browsing, and deployment
- **Claude 3.7 Integration**: Leverages Anthropic's Claude 3.7 model for reasoning and tool selection
- **LangGraph Workflow**: Uses LangGraph for state management and execution flow control
- **Interactive CLI**: Simple command-line interface for interacting with the agent

## Requirements

- Python 3.8+
- An Anthropic API key

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/manus-inspired-agent.git
cd manus-inspired-agent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Anthropic API key:
```
ANTHROPIC_API_KEY=your_api_key_here
```

## Usage

Run the agent:

```bash
python agent.py
```

The agent provides a command-line interface with these commands:
- Type your requests to interact with the agent
- Type `reset` to start a new conversation
- Type `exit` to quit

## Available Tools

The agent implementation includes the following tool categories:

### Message Tools

- **message_notify_user**: Send a message to the user without requiring a response
- **message_ask_user**: Ask the user a question and wait for a response

### File Tools

- **file_read**: Read file content
- **file_write**: Overwrite or append content to a file

### Shell Tools

- **shell_exec**: Execute commands in a specified shell session

### Browser Tools

- **browser_navigate**: Navigate browser to a specified URL

### Web Tools

- **info_search_web**: Search web pages using a search engine

### Deployment Tools

- **deploy_expose_port**: Expose a specified local port for temporary public access

## Architecture

The agent follows a loop-based architecture:

1. **Analyze Events**: Understand user needs and current state
2. **Select Tools**: Choose the next tool call based on current state
3. **Wait for Execution**: Let the tool execute and return results
4. **Iterate**: Choose one tool call per iteration and repeat steps
5. **Submit Results**: Send results to the user
6. **Enter Standby**: Enter idle state when tasks are completed

## Implementation Details

### State Management

The agent state consists of:
- Messages (conversation history)
- Current tool calls (tools selected for execution)
- Tool results (results from executed tools)
- Status (running or complete)
- Metadata (additional information)

### Tool Call Format

Tools are called using a specific format:

```
<tool_call>
tool: tool_name
params: {
  "parameter1": "value1",
  "parameter2": "value2"
}
</tool_call>
```

### Agent Workflow

The LangGraph workflow includes these nodes:
- **agent**: Calls Claude to get the next action
- **tools**: Executes the selected tools
- **complete**: Marks the task as complete

## Customization

### Adding New Tools

To add a new tool:

1. Define the tool input schema using Pydantic:
```python
class NewToolInput(BaseModel):
    """Parameters for the new_tool."""
    parameter1: str = Field(..., description="Description of parameter1")
    parameter2: Optional[int] = Field(None, description="Description of parameter2")
```

2. Implement the tool function:
```python
@tool
def new_tool(input: NewToolInput) -> str:
    """Description of what the new tool does."""
    # Tool implementation
    return f"Result from using the tool with {input.parameter1}"
```

3. Add the tool to the tools list:
```python
tools = [
    # Existing tools
    new_tool,  # Add your new tool here
]
```

### Modifying the System Prompt

The system prompt can be modified in the `SYSTEM_PROMPT` variable to adjust the agent's behavior or add specific instructions.

## Limitations and Future Work

- This implementation includes simulated tool execution for demonstration purposes
- In a real-world implementation, tools would interact with actual systems
- Future enhancements could include:
  - Web UI for easier interaction
  - More robust tool execution with error handling
  - Persistent storage for conversation history and state
  - Integration with additional LLMs or embedding models
  - Enhanced security measures for tool execution

## Credits

This implementation is inspired by the Manus agent architecture and builds upon:
- [LangGraph](https://github.com/langchain-ai/langgraph) by LangChain
- [Claude API](https://anthropic.com) by Anthropic
