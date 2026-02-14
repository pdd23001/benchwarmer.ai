
import sys
import os
import logging
from dotenv import load_dotenv

# Ensure we can import from benchwarmer package
sys.path.insert(0, ".")

# Load env vars
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

try:
    from benchwarmer.agents.intake import IntakeAgent
    print("IntakeAgent imported successfully.")
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    print(f"API Key present: {bool(api_key)}")
    if api_key:
        print(f"API Key start: {api_key[:5]}...")
    
    agent = IntakeAgent()
    print("IntakeAgent initialized.")
    
    query = "Run a minimum vertex cover benchmark on small random graphs"
    print(f"Running query: {query}")
    
    config = agent.run(query, interactive=False)
    print("Config generated successfully:")
    print(config)

except Exception as e:
    print("ERROR OCCURRED:")
    import traceback
    traceback.print_exc()
