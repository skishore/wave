#!/usr/local/bin/python3

import argparse
import http.server

class Handler(http.server.SimpleHTTPRequestHandler):
    pass

Handler.extensions_map['.wasm'] = 'application/wasm'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=8000)
    args = parser.parse_args()

    print('Serving HTTP on 0.0.0.0 port {} ...'.format(args.port))
    httpd = http.server.HTTPServer(('', args.port), Handler)
    httpd.serve_forever()
