"""
This is 'Host' script responsible for running the LLM application

Hence, Client logic is running inside of Host.
"""

from client import MCPClient
import asyncio

async def main():
    chatbot = MCPClient()
    await chatbot.connect_to_server_and_run()

if __name__ == "__main__":
    asyncio.run(main())
