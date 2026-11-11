"""Zero-dependency local HTTP server for the DNA Sequence Classification Dashboard."""
import argparse
import functools
import http.server
import os
import socketserver
import webbrowser


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve DNA Sequence Classifier Dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Port to serve on (default: 8501)")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open the browser")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=root_dir)

    with ReusableTCPServer(("127.0.0.1", args.port), handler) as httpd:
        url = f"http://127.0.0.1:{args.port}/index.html"
        print(f"Serving DNA Sequence Classification Dashboard at {url}")
        print("Press Ctrl+C to stop.")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
