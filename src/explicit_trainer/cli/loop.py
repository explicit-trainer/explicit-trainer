"""et-loop: talk to a loop served on a UNIX socket (explicit_trainer.loop.server).

et-loop --socket <path> start              the first message to the agent
et-loop --socket <path> msg <file | ->     the agent's message (from a file, or standard input) -> the answer
et-loop --socket <path> resume             after a pause: the message that resumes the agent
et-loop --socket <path> state              the run's summary, as JSON

Prints the loop's answer; exits with status 1 and the loop's error when it refuses a request.

Public components
    main()

Example
    echo "RESERVE 8 GPUs 30" | et-loop --socket run/loop.sock msg -
"""
import argparse
import json
import sys

from explicit_trainer.loop.server import request


def main():
    p = argparse.ArgumentParser(prog="et-loop")
    p.add_argument("--socket", required=True)
    p.add_argument("command", choices=["start", "msg", "resume", "state"])
    p.add_argument("file", nargs="?")
    a = p.parse_args()
    text = ""
    if a.command == "msg":
        assert a.file, "msg needs a file, or - for standard input"
        text = sys.stdin.read() if a.file == "-" else open(a.file).read()
    try:
        out = request(a.socket, a.command, text)
    except RuntimeError as e:
        sys.exit(str(e))
    print(json.dumps(out, indent=1) if isinstance(out, dict) else out)


if __name__ == "__main__":
    main()
