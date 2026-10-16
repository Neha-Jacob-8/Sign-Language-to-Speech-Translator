"""Entry point: `python server.py` (or `uvicorn server:app`, as used by the Azure App Service startup command)."""
import os

from signspeak.server import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
