import threading
import webview

from run_app import create_app


def main():
    app = create_app()
    server = app.server

    def run_server():
        server.run(port=8050, host="127.0.0.1")

    threading.Thread(target=run_server, daemon=True).start()
    webview.create_window("Cosmic Detector Analyzer", "http://127.0.0.1:8050")
    webview.start()


if __name__ == "__main__":
    main()
