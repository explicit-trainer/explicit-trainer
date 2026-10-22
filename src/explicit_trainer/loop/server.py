"""A loop served on a UNIX socket, and the client that talks to it.

Purpose and design. The process that builds a Loop (with the environment's GpuProvider and Checker)
serves it as a daemon; whoever relays the agent's messages, and the operator, talk to it with
request() or the et-loop command. One request per connection: a JSON line {"cmd": ..., "text": ...}
answered by a JSON line {"out": ...}, or {"error": ...} when the loop refuses it (e.g. resume while
running). Requests are handled one at a time, in the order they arrive, so the loop's state needs no
locking. Commands: "start", "msg" (text: the agent's message), "resume", "state" (out: the summary).

Public components
    serve(loop, socket_path)
        Serves until the process is stopped; removes a stale socket file first.
    request(socket_path, cmd, text="") -> str | dict
        Sends one request and returns "out"; raises RuntimeError with the loop's error.

Example
    serve(Loop("runs/r1", instructions, LocalGpus(), CommandChecker("...")), "runs/r1/loop.sock")
    print(request("runs/r1/loop.sock", "msg", "RESERVE 8 GPUs 30"))
"""
import json
import os
import socket
import socketserver

from explicit_trainer.loop.loop import Loop


def _handle(loop: Loop, req: dict):
    cmd = req["cmd"]
    if cmd == "start":
        return loop.start()
    if cmd == "msg":
        return loop.reply(req["text"])
    if cmd == "resume":
        return loop.resume()
    if cmd == "state":
        return loop.summary()
    raise ValueError(f"unknown command {cmd!r}")


def serve(loop: Loop, socket_path: str):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            req = json.loads(self.rfile.readline())
            try:
                answer = {"out": _handle(loop, req)}
            except (AssertionError, ValueError) as e:
                answer = {"error": f"{type(e).__name__}: {e}"}
            self.wfile.write((json.dumps(answer) + "\n").encode())

    if os.path.exists(socket_path):
        os.remove(socket_path)
    with socketserver.UnixStreamServer(socket_path, Handler) as server:
        server.serve_forever()


def request(socket_path: str, cmd: str, text: str = ""):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(socket_path)
        s.sendall((json.dumps({"cmd": cmd, "text": text}) + "\n").encode())
        answer = json.loads(s.makefile().readline())
    if "error" in answer:
        raise RuntimeError(answer["error"])
    return answer["out"]
