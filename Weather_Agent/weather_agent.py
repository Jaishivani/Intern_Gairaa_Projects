#Install all requirements
import os
import requests
from dotenv import load_dotenv
from fastmcp import FastMCP

# Load environment variables
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
if not API_KEY:
    raise ValueError("Set OPENWEATHER_API_KEY in .env")

# Create MCP server
mcp = FastMCP("Weather Bot", "Get current weather for a city")

# Define tool
@mcp.tool()
def get_weather_data(city: str) -> str:
    """Fetch current weather for a city."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    resp = requests.get(url)

    if resp.status_code != 200:
        return f"Error fetching weather: {resp.text}"
    
    data = resp.json()
    weather = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    feels = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]

    return (
        f"Weather in {city}:\n"
        f"- Condition: {weather}\n"
        f"- Temperature: {temp}°C (feels like {feels}°C)\n"
        f"- Humidity: {humidity}%"
    )

# Run MCP server
if __name__ == "__main__":
    mcp.run(transport='stdio')