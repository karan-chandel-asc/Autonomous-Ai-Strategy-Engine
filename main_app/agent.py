from langchain.agents import create_agent
from langchain_models import Lanchain_models
from langchain_core.tools import Tool
from tools import ToolClass




class AgentClass():
    def __init__(self):
        self.llm=Lanchain_models()
        self.tools=ToolClass()

    def call_agent(self,tools=[]):
        pass

    
