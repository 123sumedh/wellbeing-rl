"""
Server entry point for OpenEnv validation.
Imports and runs the main FastAPI app from the root app.py.
"""
import sys
import os
 
# Add parent directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from app import app  # noqa: F401
 
 
def main():
    """Entry point for [project.scripts] server command."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)
 
 
if __name__ == "__main__":
    main()
 
