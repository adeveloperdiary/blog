from colorama import Fore, Back, Style, init


def info_log(msg):
    # init(autoreset=True)
    print(Fore.LIGHTCYAN_EX + "[INFO]", end=" ")
    print(msg)
    # init(autoreset=True)


def output_log(msg):
    # init(autoreset=True)
    print(Fore.LIGHTYELLOW_EX + "[OUTPUT]", end=" ")
    print(msg)
    # init(autoreset=True)


def warning_log(msg):
    # init(autoreset=True)
    print(Fore.YELLOW + "[WARNING]", end=" ")
    print(msg)
    # init(autoreset=True)


def error_log(msg):
    # init(autoreset=True)
    print(Fore.LIGHTRED_EX + "[ERROR]", end=" ")
    print(msg)
    # init(autoreset=True)
