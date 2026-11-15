import os
import socket
import uvicorn
from app.main import app

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    default_port = int(os.environ.get("PORT", 8081))
    port = default_port

    if is_port_in_use(port, host):
        alt_port = 8082 if port == 8081 else port + 1
        print(f"[RetaillySale] Port {port} is currently in use. Automatically binding to port {alt_port}.")
        port = alt_port

    print(f"[RetaillySale] Starting development server on http://{host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True)
