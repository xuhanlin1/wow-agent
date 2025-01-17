"""查询天气（虚拟天气）"""

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_model import llm

def get_weather(city: str) -> int:
    """
    Gets the weather temperature of a specified city.

    Args:
    city (str): The name or abbreviation of the city.

    Returns:
    int: The temperature of the city. Returns 20 for 'NY' (New York),
         30 for 'BJ' (Beijing), and -1 for unknown cities.
    """

    # Convert the input city to uppercase to handle case-insensitive comparisons
    city = city.upper()

    # Check if the city is New York ('NY')
    if city == "NY":
        return 20  # Return 20°C for New York

    # Check if the city is Beijing ('BJ')
    elif city == "BJ":
        return 30  # Return 30°C for Beijing

    # If the city is neither 'NY' nor 'BJ', return -1 to indicate unknown city
    else:
        return -1

weather_tool = FunctionTool.from_defaults(fn=get_weather)

agent = ReActAgent.from_tools([ weather_tool], llm=llm, verbose=True)

response = agent.chat("纽约天气怎么样?")
