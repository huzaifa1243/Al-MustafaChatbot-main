from waitress import serve
from chat import app

if __name__ == "__main__":
    serve(app, host='127.0.0.1', port=5000, threads=8, connection_limit=1000)
