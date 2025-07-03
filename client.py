"""
This is 'Client' script responsible for creating and maintaining a 1:1 connection with server

This client script is supposed to connect to server named 'server.py'
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List
import mcp.types as types
import asyncio

class SearchQuery(BaseModel):
    query_term: str = Field(description="A single search query which will be used to search for research papers based on user's input, could contain id also")

class MCPClient:
    def __init__(self):
        self.session: ClientSession = None
        self.model_with_tools = None
        self.model = ChatOllama(model="qwen2.5:3b", temperature=0)
        self.model_with_structured_output = self.model.with_structured_output(SearchQuery)
        self.available_tools: List[dict] = []
        self.available_prompts: List[dict] = []

    def _build_prompt(self, prompt: List[types.GetPromptResult], user_query: str):
        system_prompt = prompt.messages[0].content.text
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query)
        ]

    async def _query_from_user(self, query):
        if not self.model_with_tools:
            self.model_with_tools = self.model.bind_tools(self.available_tools)

        query_prompt = await self.session.get_prompt(name="query_creater")
        print(self._build_prompt(query_prompt, query))
        response = self.model_with_structured_output.invoke(
            self._build_prompt(query_prompt, query)
        )

        print(response.query_term)

        tool_prompt = await self.session.get_prompt(name="tool_creater")
        answer = self.model_with_tools.invoke(
            self._build_prompt(tool_prompt, response.query_term)
        )

        print(answer)

        if isinstance(answer, AIMessage) and answer.tool_calls:
            for tool in answer.tool_calls:
                tool_name = tool["name"]
                tool_args = tool["args"]

                print(f"Calling tool {tool_name} with args {tool_args}")

                results = await self.session.call_tool(tool_name, arguments=tool_args)
                if results.content:
                    answer_prompt = await self.session.get_prompt(name="answer_creater", arguments={"result": " ID: ".join([result.text for result in results.content])})
                    response = self.model.invoke(
                        self._build_prompt(answer_prompt, query)
                    )

                    print(f"LLM Output: \n{response.content}")
                else:
                    print("Nothing to output! All these pdfs already available.")

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
        
                if query.lower() == 'quit':
                    break
                    
                await self._query_from_user(query)
                print("\n")
                    
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def connect_to_server_and_run(self):
        server_params = StdioServerParameters(
            command="uv", 
            args=["run", "server.py"],  
            env=None,  
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                await self.session.initialize()

                response_tools = await self.session.list_tools()
                response_prompts = await self.session.list_prompts()
                
                tools = response_tools.tools
                prompts = response_prompts.prompts

                print("Connected to server!")
                print("\nAvailable tools: ", [tool.name for tool in tools])
                print("\nAvailable prompts: ", [prompt.name for prompt in prompts])
                
                self.available_tools = [{
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                } for tool in response_tools.tools]
    
                self.available_prompts = [{
                    "name": prompt.name,
                    "description": prompt.description,
                } for prompt in response_prompts.prompts]

                await self.chat_loop()

