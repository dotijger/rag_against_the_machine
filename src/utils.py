
def read_source(path: str) -> str:
    """
    Reads the file path and returns a string of its contents in utf-8 encoding

    Arg:
     - path (file path) as string

    Returns:
     - the contents of the file path as string, skipping \n characters

    """
    with open(path, encoding="utf-8", newline="") as file:
        return file.read()
