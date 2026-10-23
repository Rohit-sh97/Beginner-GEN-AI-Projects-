import sys


def print_status(message: str) -> None:
    stream = sys.stdout
    encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")

    if encoding != "utf8" and hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        stream = sys.stdout

    try:
        print(message, file=stream, flush=True)
    except UnicodeEncodeError:
        safe_message = message.encode(
            getattr(stream, "encoding", None) or "ascii",
            errors="backslashreplace"
        ).decode(getattr(stream, "encoding", None) or "ascii")
        print(safe_message, file=stream, flush=True)
