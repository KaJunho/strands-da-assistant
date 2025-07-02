# Data Analysis Agent 
Data analysis assistant using Amazon Bedrock Knowledge Base and Guardrail, based on Strands SDK.
This project is developed upon https://github.com/jief123/strands-web-ui/

## Prerequisites
- Python 3.10 or higher
- Streamlit
- Strands Agents SDK
- MCP


## How to use?

```bash
# Clone the repository
git clone https://github.com/KaJunho/strands-da-assistant.git

# cd to strands-da-assistant/, create a python virtual environment called .venv.
cd strands-da-assistant
python -m venv .venv
source .venv/bin/activate

cd strands-web-ui

# Install in development mode
pip install -e .
```
First, cd to strands-da-assistant/, create a python virtual environment called .venv. <br/>
Next, follow the instructions below in **Installation section**, to install the project. <br/>
Then, cd to strands-da-assistant/src/strands_web_ui/ , add a .env file to configure environment variables. <br/>

.env should be like:
```bash
aws_access_key_id = <your ak here>
aws_secret_access_key = <your sk here>
knowledge_base_id = <your kb id here>
guardrail_id = <your guardrail id here>
guardrail_version = <version>
```

Note: for knowledge base and guardrail creation, you can deploy the cloudformation template (game-kb-with-guardrail.yaml).

Finally, cd to strands-da-assistant/, run `bash start_app.sh` in your console.

## Test questions
1. 公司有多少名员工
2. 请告诉我2024年销售额最高员工的姓名，员工ID，生日，性别，婚姻状况，岗位
3. 哪名员工2024年、2025年游戏总销售量最多？
4. 请查询2024年和2025年的总销量，概括公司销量的趋势
5. 请概括公司不同division，2024至2025年的销量趋势


--------------------
# Strands Web UI

A Streamlit-based web interface for Strands Agents with thinking process visualization and MCP integration.

## Features

- 🤖 Interactive chat interface with streaming responses
- 💭 Visualization of agent thinking process
- 🔧 Tool execution and result display
- 🔌 MCP server integration for extended capabilities
- ⚙️ Configurable model and agent parameters
- 💬 Conversation history management

## Installation

### Prerequisites

- Python 3.10 or higher
- [Streamlit](https://streamlit.io/)
- [Strands Agents SDK](https://github.com/strands-agents/sdk-python)
- [MCP](https://github.com/model-context-protocol/mcp)

### Installation

#### Install from PyPI (Coming Soon)

```bash
pip install strands-web-ui
```

#### Install from Source (Development Mode)

```bash
# Clone the repository
git clone https://github.com/jief123/strands-web-ui.git
cd strands-web-ui

# Install in development mode
pip install -e .
```

This will install the package in "editable" mode, which means you can modify the source code and see the changes immediately without reinstalling.

## Usage

### Quick Start

After installation, you can run the application in several ways:

```bash
# Method 1: Run directly with Streamlit
streamlit run app.py

# Method 2: Run the basic chat example
python examples/basic_chat.py

# Method 3: Run the custom tools example
python examples/custom_tools.py
```

You can also import and use the package in your own Python scripts:

```python
import streamlit as st
from strands_web_ui.app import main

# Run the Strands Web UI application
main()
```

## Configuration

The application can be configured through JSON files in the `config` directory:

- `config_with_thinking.json`: Main configuration file with model, agent, and UI settings
- `mcp_config.json`: MCP server configuration
- `example_config.json`: Example configuration with different settings

### Model Settings

Configure the model provider, model ID, region, and token limits:

```json
"model": {
    "provider": "bedrock",
    "model_id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "region": "us-east-1",
    "max_tokens": 24000
}
```

### Agent Settings

Configure the system prompt, tool execution, and thinking capabilities:

```json
"agent": {
    "system_prompt": "You are a helpful assistant that provides concise, accurate information.",
    "max_parallel_tools": 4,
    "record_direct_tool_call": true,
    "hot_reload_tools": true,
    "enable_native_thinking": true,
    "thinking_budget": 16000
}
```

### Conversation Settings

Adjust history window size and management:

```json
"conversation": {
    "window_size": 20,
    "summarize_overflow": true
}
```

### UI Settings

Configure UI update intervals:

```json
"ui": {
    "update_interval": 0.1
}
```

## MCP Server Integration

To use MCP servers:

1. Configure servers in `config/mcp_config.json`:

```json
{
  "mcpServers": {
    "server-id": {
      "command": "command-to-run",
      "args": ["arg1", "arg2"],
      "env": {"ENV_VAR": "value"}
    }
  }
}
```

2. Connect to servers through the UI
3. Use the tools provided by the servers in your conversations

## Project Structure

```
strands_web_ui/
├── config/                  # Configuration files
│   ├── config_with_thinking.json
│   ├── mcp_config.json
│   └── example_config.json
├── examples/                # Example scripts
│   ├── basic_chat.py
│   └── custom_tools.py
├── src/                     # Source code
│   └── strands_web_ui/
│       ├── __init__.py
│       ├── app.py           # Main application
│       ├── mcp_server_manager.py
│       ├── handlers/        # Event handlers
│       │   └── streamlit_handler.py
│       └── utils/           # Utility functions
│           └── config_loader.py
├── static/                  # Static assets
├── tests/                   # Tests
├── LICENSE
├── README.md
├── CONTRIBUTING.md
└── pyproject.toml
```

## How It Works

This application demonstrates how to:

1. Create a Strands agent with thinking capabilities
2. Connect to MCP servers for extended functionality
3. Visualize the agent's thinking process in real-time
4. Maintain conversation history across interactions
5. Execute tools and display results

The thinking process visualization shows how the agent reasons through problems, making the decision-making process transparent to users.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
