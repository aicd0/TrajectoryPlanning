import platform

def is_windows() -> bool:
    return platform.system() == "Windows"

def is_linux() -> bool:
    return platform.system() == "Linux"

def check_platform() -> None:
    if not is_windows() and not is_linux():
        raise Exception('Platform not supported.')