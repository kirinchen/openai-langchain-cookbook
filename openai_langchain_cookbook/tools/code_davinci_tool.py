from typing import Optional
from math import sqrt, cos, sin

from langchain_core.tools import BaseTool

desc = """
    
    """


class CodeDavinciTool(BaseTool):
    name = "Hypotenuse calculator"
    description = desc

    def _run(self, query: str = None, ):
        if query is None:
            return "no query"

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
