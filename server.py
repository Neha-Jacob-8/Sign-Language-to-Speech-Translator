"""Entry point for the optional local server: `python server.py` (or `uvicorn server:app`)."""
import os

from signspeak.server import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
