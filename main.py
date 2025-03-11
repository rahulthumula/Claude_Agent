import os
import json
import re
import time
from typing import Dict, List, Any, Optional, TypedDict, Literal, Union
from datetime import datetime
import subprocess
from pathlib import Path

# LangGraph and LangChain imports
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ===== State Management =====

class MessagesState(TypedDict):
    """State for the agent conversation, including messages and execution status."""
    messages: List[Union[SystemMessage, HumanMessage, AIMessage, ToolMessage]]
    current_tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    status: Literal["RUNNING", "COMPLETE"]
    metadata: Dict[str, Any]

# ===== System Prompt =====

SYSTEM_PROMPT = """
You are Manus, an AI agent created to help users accomplish tasks.

You excel at the following tasks:
1. Information gathering, fact-checking, and documentation
2. Data processing, analysis, and visualization
3. Writing multi-chapter articles and in-depth research reports
4. Creating websites, applications, and tools
5. Using programming to solve various problems beyond development
6. Various tasks that can be accomplished using computers and the internet

Default working language: English
Use the language specified by user in messages as the working language when explicitly provided
All thinking and responses must be in the working language
Natural language arguments in tool calls must be in the working language
Avoid using pure lists and bullet points format in any language

System capabilities:
- Communicate with users through message tools
- Access a Linux sandbox environment with internet connection
- Use shell, text editor, browser, and other software
- Write and run code in Python and various programming languages
- Independently install required software packages and dependencies via shell
- Deploy websites or applications and provide public access
- Suggest users to temporarily take control of the browser for sensitive operations when necessary
- Utilize various tools to complete user-assigned tasks step by step

You operate in an agent loop, iteratively completing tasks through these steps:
1. Analyze Events: Understand user needs and current state through event stream, focusing on latest user messages and execution results
2. Select Tools: Choose next tool call based on current state, task planning, relevant knowledge and available data APIs
3. Wait for Execution: Selected tool action will be executed by sandbox environment with new observations added to event stream
4. Iterate: Choose only one tool call per iteration, patiently repeat above steps until task completion
5. Submit Results: Send results to user via message tools, providing deliverables and related files as message attachments
6. Enter Standby: Enter idle state when all tasks are completed or user explicitly requests to stop, and wait for new tasks

When you need to use a tool, follow this exact format:
<tool_call>
tool: tool_name
params: {
  "parameter1": "value1",
  "parameter2": "value2"
}
</tool_call>

Here are the available tools:
{tool_descriptions}
"""

# ===== Tool Definitions =====

# --- Message Tools ---
class MessageNotifyUserInput(BaseModel):
    """Parameters for the message_notify_user tool."""
    text: str = Field(..., description="Message text to display to user")
    attachments: Optional[Union[str, List[str]]] = Field(None, description="(Optional) List of attachments to show to user, can be file paths or URLs")

@tool
def message_notify_user(input: MessageNotifyUserInput) -> str:
    """Send a message to user without requiring a response. Use for acknowledging receipt of messages, providing progress updates, reporting task completion, or explaining changes in approach."""
    attachment_str = ""
    if input.attachments:
        if isinstance(input.attachments, str):
            attachment_str = f"\nAttachment: {input.attachments}"
        else:
            attachment_str = f"\nAttachments: {', '.join(input.attachments)}"
    
    return f"Message sent to user: {input.text}{attachment_str}"

class MessageAskUserInput(BaseModel):
    """Parameters for the message_ask_user tool."""
    text: str = Field(..., description="Question text to present to user")
    attachments: Optional[Union[str, List[str]]] = Field(None, description="(Optional) List of question-related files or reference materials")
    suggest_user_takeover: Optional[Literal["none", "browser"]] = Field("none", description="(Optional) Suggested operation for user takeover")

@tool
def message_ask_user(input: MessageAskUserInput) -> str:
    """Ask user a question and wait for response. Use for requesting clarification, asking for confirmation, or gathering additional information."""
    # For simplicity in this demo, we'll just return a simulated response
    # In a real implementation, this would wait for actual user input
    
    takeover_str = ""
    if input.suggest_user_takeover and input.suggest_user_takeover != "none":
        takeover_str = f"\nSuggested takeover: {input.suggest_user_takeover}"
    
    attachment_str = ""
    if input.attachments:
        if isinstance(input.attachments, str):
            attachment_str = f"\nAttachment: {input.attachments}"
        else:
            attachment_str = f"\nAttachments: {', '.join(input.attachments)}"
    
    return f"User was asked: {input.text}{attachment_str}{takeover_str}\nUser response: [Simulated user response for demonstration]"

# --- File Tools ---
class FileReadInput(BaseModel):
    """Parameters for the file_read tool."""
    file: str = Field(..., description="Absolute path of the file to read")
    start_line: Optional[int] = Field(None, description="(Optional) Starting line to read from, 0-based")
    end_line: Optional[int] = Field(None, description="(Optional) Ending line number (exclusive)")
    sudo: Optional[bool] = Field(False, description="(Optional) Whether to use sudo privileges")

@tool
def file_read(input: FileReadInput) -> str:
    """Read file content. Use for checking file contents, analyzing logs, or reading configuration files."""
    # Simulated file system for demonstration
    simulation_files = {
        "/home/user/readme.txt": "This is a sample README file.\nIt contains information about the project.",
        "/home/user/data.csv": "id,name,value\n1,Item1,10.5\n2,Item2,20.3\n3,Item3,15.7",
        "/home/user/config.json": '{"api_key": "sample_key", "max_tokens": 4096, "temperature": 0.7}'
    }
    
    if input.file in simulation_files:
        content = simulation_files[input.file]
        lines = content.split('\n')
        
        start = input.start_line if input.start_line is not None else 0
        end = input.end_line if input.end_line is not None else len(lines)
        
        return '\n'.join(lines[start:end])
    else:
        return f"Error: File '{input.file}' not found. Available files: {list(simulation_files.keys())}"

class FileWriteInput(BaseModel):
    """Parameters for the file_write tool."""
    file: str = Field(..., description="Absolute path of the file to write to")
    content: str = Field(..., description="Text content to write")
    append: Optional[bool] = Field(False, description="(Optional) Whether to use append mode")
    leading_newline: Optional[bool] = Field(False, description="(Optional) Whether to add a leading newline")
    trailing_newline: Optional[bool] = Field(False, description="(Optional) Whether to add a trailing newline")
    sudo: Optional[bool] = Field(False, description="(Optional) Whether to use sudo privileges")

@tool
def file_write(input: FileWriteInput) -> str:
    """Overwrite or append content to a file. Use for creating new files, appending content, or modifying existing files."""
    # For demonstration, we'll simulate writing to a file
    mode = "append" if input.append else "overwrite"
    content = input.content
    
    if input.leading_newline:
        content = "\n" + content
    
    if input.trailing_newline:
        content = content + "\n"
    
    return f"File '{input.file}' {mode}d with {len(content)} characters of content."

# --- Shell Tools ---
class ShellExecInput(BaseModel):
    """Parameters for the shell_exec tool."""
    id: str = Field(..., description="Unique identifier of the target shell session")
    exec_dir: str = Field(..., description="Working directory for command execution (must use absolute path)")
    command: str = Field(..., description="Shell command to execute")

@tool
def shell_exec(input: ShellExecInput) -> str:
    """Execute commands in a specified shell session. Use for running code, installing packages, or managing files."""
    # For demonstration, we'll simulate shell command execution
    
    # Simulated commands and responses
    command_responses = {
        "ls": "file1.txt\nfile2.py\nfolder1/\nfolder2/",
        "echo": lambda args: args,
        "python": lambda args: f"Executed Python with arguments: {args}",
        "cat": lambda args: f"Contents of {args} would be shown here",
        "pip": lambda args: f"Pip executed with arguments: {args}"
    }
    
    # Parse the command
    parts = input.command.split(" ", 1)
    cmd = parts[0]
    args = parts[1] if len(parts) > 1 else ""
    
    # Simulate command execution
    if cmd in command_responses:
        if callable(command_responses[cmd]):
            response = command_responses[cmd](args)
        else:
            response = command_responses[cmd]
        
        return f"Shell session '{input.id}' executed '{input.command}' in '{input.exec_dir}':\n{response}"
    else:
        return f"Shell session '{input.id}' executed '{input.command}' in '{input.exec_dir}':\nCommand simulated (no specific mock response for '{cmd}')"

# --- Browser Tools ---
class BrowserNavigateInput(BaseModel):
    """Parameters for the browser_navigate tool."""
    url: str = Field(..., description="Complete URL to visit. Must include protocol prefix.")

@tool
def browser_navigate(input: BrowserNavigateInput) -> str:
    """Navigate browser to specified URL. Use when accessing new pages is needed."""
    # Simulate browser navigation
    return f"Browser navigated to: {input.url}\nPage title: Simulated Page Title\nContent snippet: This is a simulated representation of the page content..."

class InfoSearchWebInput(BaseModel):
    """Parameters for the info_search_web tool."""
    query: str = Field(..., description="Search query in Google search style, using 3-5 keywords")
    date_range: Optional[Literal["all", "past_hour", "past_day", "past_week", "past_month", "past_year"]] = Field("all", description="(Optional) Time range filter for search results")

@tool
def info_search_web(input: InfoSearchWebInput) -> str:
    """Search web pages using search engine. Use for obtaining latest information or finding references."""
    # Simulate web search
    date_range_str = f" (filtered to {input.date_range})" if input.date_range != "all" else ""
    
    return f"Search results for '{input.query}'{date_range_str}:\n\n1. Result 1: {input.query} - Information...\n2. Result 2: More about {input.query}...\n3. Result 3: Related topic to {input.query}..."

# --- Deployment Tools ---
class DeployExposePortInput(BaseModel):
    """Parameters for the deploy_expose_port tool."""
    port: int = Field(..., description="Local port number to expose")

@tool
def deploy_expose_port(input: DeployExposePortInput) -> str:
    """Expose specified local port for temporary public access. Use when providing temporary public access for services."""
    # Simulate port exposure
    simulated_url = f"https://exposed-{input.port}.example.com"
    return f"Port {input.port} exposed successfully!\nPublic URL: {simulated_url}\nURL will remain active for the next 2 hours."

# Collect all tools
tools = [
    message_notify_user,
    message_ask_user,
    file_read,
    file_write,
    shell_exec,
    browser_navigate,
    info_search_web,
    deploy_expose_port
]

# Create tool descriptions for the system prompt
def get_tool_descriptions() -> str:
    """Generate formatted tool descriptions for the system prompt."""
    descriptions = []
    for tool in tools:
        tool_name = tool.name
        tool_description = tool.description
        parameters = []
        
        # Get parameters from tool schema
        try:
            schema = tool.args_schema.schema()
            if 'properties' in schema:
                for param_name, param_details in schema['properties'].items():
                    required = param_name in schema.get('required', [])
                    param_type = param_details.get('type', 'any')
                    param_desc = param_details.get('description', 'No description')
                    param_str = f"  - {param_name} ({param_type}, {'Required' if required else 'Optional'}): {param_desc}"
                    parameters.append(param_str)
        except:
            # Fallback if schema extraction fails
            parameters.append("  [Parameters could not be extracted]")
        
        # Format the tool description
        description = f"Tool: {tool_name}\nDescription: {tool_description}\nParameters:\n" + "\n".join(parameters)
        descriptions.append(description)
    
    return "\n\n".join(descriptions)

# ===== Agent Implementation =====

# Format the system prompt with tool descriptions
system_prompt = SYSTEM_PROMPT.format(tool_descriptions=get_tool_descriptions())

# Initialize the Claude 3.7 model with tools
model = ChatAnthropic(
    model="claude-3-7-sonnet-20240307",
    temperature=0.2,
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
).bind_tools(tools)

# Initialize tool node
tool_node = ToolNode(tools)

# Tool call extraction from Claude's response
def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract tool calls from Claude's response."""
    # Pattern to match tool calls in the format specified in the system prompt
    pattern = r'<tool_call>\s*tool:\s*(\w+)\s*params:\s*(\{.*?\})\s*</tool_call>'
    matches = re.finditer(pattern, text, re.DOTALL)
    
    tool_calls = []
    for match in matches:
        tool_name = match.group(1).strip()
        params_str = match.group(2).strip()
        
        try:
            # Parse the JSON parameters
            params = json.loads(params_str)
            tool_calls.append({
                "name": tool_name,
                "parameters": params
            })
        except json.JSONDecodeError:
            # If JSON parsing fails, try a simpler key-value extraction
            params = {}
            # Simple key-value extraction (this is a fallback and not perfect)
            param_lines = params_str.strip('{}').split('\n')
            for line in param_lines:
                if ':' in line:
                    parts = line.split(':', 1)
                    key = parts[0].strip().strip('"\'')
                    value = parts[1].strip().strip(',').strip().strip('"\'')
                    params[key] = value
            
            tool_calls.append({
                "name": tool_name,
                "parameters": params
            })
    
    return tool_calls

# Function to call the model and extract tool calls
def call_model(state: MessagesState) -> MessagesState:
    """Call the model and extract tool calls from its response."""
    messages = state["messages"]
    
    # If this is the first run and no system message, add it
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [SystemMessage(content=system_prompt)] + messages
    
    # Call the model
    try:
        response = model.invoke(messages)
        
        # Extract tool calls from the response
        content = response.content
        tool_calls = extract_tool_calls(content)
        
        # Return updated state
        return {
            "messages": state["messages"] + [response],
            "current_tool_calls": tool_calls,
            "tool_results": state["tool_results"],
            "status": state["status"],
            "metadata": state["metadata"]
        }
    except Exception as e:
        # Handle errors
        error_message = f"Error calling model: {str(e)}"
        return {
            "messages": state["messages"] + [AIMessage(content=f"I encountered an error: {error_message}. Let me try again.")],
            "current_tool_calls": [],
            "tool_results": state["tool_results"],
            "status": state["status"],
            "metadata": state["metadata"]
        }

# Define routing logic for the agent
def should_continue(state: MessagesState) -> Literal["tools", "complete"]:
    """Determine whether to use tools or complete the interaction."""
    if state["current_tool_calls"]:
        return "tools"
    
    # Check if the last message indicates task completion
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and "task_complete" in state["metadata"] and state["metadata"]["task_complete"]:
        return "complete"
    
    # Check for explicit task completion phrases in the last message
    completion_phrases = [
        "task is complete",
        "task has been completed",
        "completed all tasks",
        "finished the task",
        "i've completed",
        "i have completed"
    ]
    
    if isinstance(last_message, AIMessage) and any(phrase in last_message.content.lower() for phrase in completion_phrases):
        return "complete"
    
    return "tools"

# Function to mark task as complete
def complete_task(state: MessagesState) -> MessagesState:
    """Mark the task as complete."""
    new_state = state.copy()
    new_state["status"] = "COMPLETE"
    new_state["metadata"]["task_complete"] = True
    return new_state

# Create the agent graph
def create_agent_workflow():
    """Create and configure the agent workflow."""
    # Initialize the graph
    workflow = StateGraph(MessagesState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_node("complete", complete_task)
    
    # Add edges
    workflow.add_edge(START, "agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": lambda state: state["current_tool_calls"],
            "complete": lambda state: not state["current_tool_calls"]
        }
    )
    
    workflow.add_edge("tools", "agent")
    workflow.add_edge("complete", END)
    
    # Initialize memory for state persistence
    checkpointer = MemorySaver()
    
    # Compile the graph
    return workflow.compile(checkpointer=checkpointer)

# ===== Chat Interface =====

def chat_interface():
    """Interactive command-line interface for the agent."""
    # Create the agent
    agent = create_agent_workflow()
    
    print("\n🤖 Manus-Inspired Agent initialized")
    print("Type 'exit' to quit, 'reset' to start a new conversation")
    
    # Initial state
    state = {
        "messages": [SystemMessage(content=system_prompt)],
        "current_tool_calls": [],
        "tool_results": [],
        "status": "RUNNING",
        "metadata": {"task_complete": False}
    }
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() == "exit":
            print("🤖 Goodbye!")
            break
            
        if user_input.lower() == "reset":
            print("🤖 Starting a new conversation")
            state = {
                "messages": [SystemMessage(content=system_prompt)],
                "current_tool_calls": [],
                "tool_results": [],
                "status": "RUNNING",
                "metadata": {"task_complete": False}
            }
            continue
        
        # Add user message
        state["messages"].append(HumanMessage(content=user_input))
        
        # Show thinking indicator
        print("\nAgent: thinking", end="")
        for _ in range(3):
            time.sleep(0.5)
            print(".", end="", flush=True)
        print("\r" + " " * 20 + "\r", end="")
        
        # Invoke the agent
        try:
            final_state = agent.invoke(state)
            
            # Update state
            state = final_state
            
            # Print the agent's response
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage):
                # Clean up tool call syntax for display purposes
                display_content = re.sub(r'<tool_call>.*?</tool_call>', '[Tool Call]', last_message.content, flags=re.DOTALL)
                print(f"Agent: {display_content}")
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

# ===== Main Function =====

def main():
    """Main entry point."""
    # Check if API key is set
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not found in environment variables.")
        print("Please set your API key in a .env file or environment variables.")
        return
        
    # Run the chat interface
    chat_interface()

if __name__ == "__main__":
    main()