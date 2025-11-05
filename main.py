from dotenv import load_dotenv

load_dotenv()

from datetime import datetime
import logging
from utils import to_json_str
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage

SYSTEM_PROMPT = """
You are a helpful computer assistant that have access to the internet, local file system, and python runtime.
Keep in mind that:
    - Do not be overly confident in your own knowledge. Searching can provide a broader perspective and help validate existing knowledge.  
    - In your response, you should try to refer to the materials you found as much as possible. Also add urls for references.
    - If you have enough information, you should answer the question directly instead of searching again.

Today is {}
"""

class ComputerAgent:
    def __init__(self, tools, max_num_hops=3):
        self.tools = tools
        model = init_chat_model(
            model='gemini-2.5-flash',
            model_provider="google_genai",
        )
        self.model = model.bind_tools(tools)
        self.max_num_hops = max_num_hops
        self.messages = [] # history of actions
        self.tool_map = {tool.name: tool for tool in tools}
        self.logger = logging.getLogger(__name__)
        
        # Configure logging if not already configured
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

    
    async def run(self, query: str) -> str:
        self.logger.info(f"🚀 Starting agent with query: {query}")
        self.messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(datetime.now().strftime("%Y-%m-%d"))),
            HumanMessage(content=query)
        ]
        
        hops = 0
        while hops < self.max_num_hops:
            self.logger.info(f"🔄 Starting hop {hops + 1}/{self.max_num_hops}")
            
            try:
                response: AIMessage = self.model.invoke(self.messages) # type: ignore
                self.messages.append(response)
                
                if not response.tool_calls:
                    self.logger.info("✅ LLM concluded")
                    return response.content # type: ignore
                
                tool_results = []
                # may generate multiple tool calls in one response
                for i, tool_call in enumerate(response.tool_calls):
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]
                    
                    self.logger.info(f"🛠️  Tool call {i+1}: {tool_name} with args: {tool_args}")
                    
                    if tool_name in self.tool_map:
                        try:
                            result = await self.tool_map[tool_name].ainvoke(tool_args)
                            result_str = to_json_str(result, indent=2)
                            
                            result_preview = result_str[:1000] + '...' if len(result_str) > 1000 else result_str
                            self.logger.info(f"✅ Tool {tool_name} result: {result_preview}")
                            
                            tool_message = ToolMessage(
                                content=result_str,
                                tool_call_id=tool_id
                            )
                            tool_results.append(tool_message)
                            
                        except Exception as e:
                            self.logger.error(f"❌ Error executing {tool_name}: {str(e)}")
                            error_message = ToolMessage(
                                content=f"Error executing {tool_name}: {str(e)}",
                                tool_call_id=tool_id
                            )
                            tool_results.append(error_message)
                    else:
                        self.logger.error(f"❌ Tool {tool_name} not found in available tools")
                        error_message = ToolMessage(
                            content=f"Tool {tool_name} not found",
                            tool_call_id=tool_id
                        )
                        tool_results.append(error_message)
                
                self.messages.extend(tool_results)
                hops += 1
                self.logger.info(f"📝 Added {len(tool_results)} tool result(s) to conversation")
                
            except Exception as e:
                self.logger.error(f"💥 Error during hop {hops + 1}: {str(e)}")
                return f"Error occurred during search: {str(e)}"
        
        self.logger.info(f"🏁 Reached max hops ({self.max_num_hops}), generating final response...")
        
        # Add instruction to generate final response based on gathered information
        final_instruction = HumanMessage(content="Please provide your answer to the original query based on all the information gathered so far.")
        self.messages.append(final_instruction)
        
        try:
            final_response = self.model.invoke(self.messages)
            self.logger.info("✅ Final response generated successfully")
            return final_response.content
        except Exception as e:
            self.logger.error(f"❌ Error generating final response: {str(e)}")
            return f"Error generating final response: {str(e)}"

import asyncio
from search import tavily_search

def main():
    tools = [tavily_search]
    agent = ComputerAgent(tools=tools, max_num_hops=3)
    query = "Using the most recent 10-Q reports in 2025, which line of item contribute to the most expense for Alphabet Inc ?"
    print("="*80)
    print(f"Running agent with query: {query}")
    print("="*80)
    
    result = asyncio.run(agent.run(query))
    print("Final Answer:")
    print(result)

if __name__ == "__main__":
    main()
