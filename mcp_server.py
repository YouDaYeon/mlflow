from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
import os

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv()

client = MultiServerMCPClient(
    {
        "mlflow-mcp": {
            "transport": "stdio",
            "command": "uv",
            "args": ["run", "--with", "mlflow[mcp]>=3.5.1", "mlflow", "mcp", "run"],
            "env": {
                "MLFLOW_TRACKING_URI": "http://localhost:5000"
            }
        }
    }
)
async def main():
    
    tools = await client.get_tools()
    agent = create_agent("openai:gpt-4.1", tools)

    system_prompt = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You can use the MLflow MCP server to get information about the MLflow experiments."
        }
    ]

    response = await agent.ainvoke({"messages": system_prompt + [HumanMessage(content="Find latest evaluation run in experiment 'ultralytics-yolo'.")]})

    for msg in response['messages']:
        who = msg.type
        content = msg.content if msg.content != '' else msg.lc_attributes

        print(f"{who}: {content}")

if __name__ == "__main__":
    asyncio.run(main())