from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List
import asyncio
# import nest_asyncio

class SearchQuery(BaseModel):
    query_term: str = Field(description="A single search query which will be used to search for research papers based on user's input, could contain id also")

class MCPClient:
    def __init__(self):
        self.session: ClientSession = None
        self.model_with_tools = None
        self.model = ChatOllama(model="llama3.2:3b", temperature=0)
        self.model_with_structured_output = self.model.with_structured_output(SearchQuery)
        self.available_tools: List[dict] = []

    async def _query_from_user(self, query):
        if not self.model_with_tools:
            self.model_with_tools = self.model.bind_tools(self.available_tools)

        response = self.model_with_structured_output.invoke(
        [
            {
                "role": "system",
                "content": """
You are a helpful bot, whose task is to create a natural language search term for searching research paper for user.
If the user mentions description of research paper, understand the description and based on that, output a single sentence search term.
If the user asks to search paper based on id, you must EXPLICITLY tell in natural language form to search on that id, YOU MUST NOT OUTPUT ID ALONE!

For example:
user: Can you help me get papers that are published for knee othroscopy
bot: Knee othroscopy papers

user: Can you find paper with id abc123
bot: Search paper with id abc123
"""
            },
            {
                "role": "user",
                "content": query
            }
            ]
        )

        print(response.query_term)

        answer = self.model_with_tools.invoke(
            response.query_term
        )

        print(answer)

        if isinstance(answer, AIMessage) and answer.tool_calls:
            for tool in answer.tool_calls:
                tool_name = tool["name"]
                tool_args = tool["args"]

                print(f"Calling tool {tool_name} with args {tool_args}")

                result = await self.session.call_tool(tool_name, arguments=tool_args)
                print(result)
                response = self.model.invoke(
                [
                    {
                "role": "system",
                "content": f"""
You are a helpful bot whose task is give a natural language response to user based on output from the tool call and user question.
The tool output will either be a list of ids for research paper or information about research papers.
In both cases, you just need to rephrase user question and output Tool Call results.
You must not output anything else apart from rephrasing user question and outputing tool call results.


Tool call result:
{str(result.content)}
"""
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                    ]
                )

                print(f"LLM Output: \n{response.content}")

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

                response = await self.session.list_tools()
                
                tools = response.tools
                print("\nConnected to server with tools:", [tool.name for tool in tools])
                
                self.available_tools = [{
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                } for tool in response.tools]
    
                await self.chat_loop()

async def main():
    chatbot = MCPClient()
    await chatbot.connect_to_server_and_run()

if __name__ == "__main__":
    asyncio.run(main())