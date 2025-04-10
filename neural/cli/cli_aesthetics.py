"""
CLI aesthetics module for Neural.
Provides ASCII art, loading animations, and other visual enhancements for the CLI.
"""

import time
import sys
import threading
import random
from typing import List, Optional, Callable

# Neural ASCII Logo
NEURAL_LOGO = """
███╗   ██╗███████╗██╗   ██╗██████╗  █████╗ ██╗
████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗██║
██╔██╗ ██║█████╗  ██║   ██║██████╔╝███████║██║
██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══██║██║
██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║███████╗
╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
"""

# Smaller Neural ASCII Logo
NEURAL_LOGO_SMALL = """
 _   _                      _
| \ | | ___ _   _ _ __ __ _| |
|  \| |/ _ \ | | | '__/ _` | |
| |\  |  __/ |_| | | | (_| | |
|_| \_|\___|\__,_|_|  \__,_|_|
"""

# Neural Network ASCII Art
NEURAL_NETWORK_ART = """
   [Input]      [Hidden]     [Output]
     ○             ○            ○
     ○           ○ ○ ○          ○
     ○             ○            ○
     ○           ○ ○ ○          ○
     ○             ○            ○
"""

# Command headers
COMMAND_HEADERS = {
    "visualize": """
╔═══════════════════════════════════════════╗
║           Neural Visualization            ║
╚═══════════════════════════════════════════╝
""",
    "compile": """
╔═══════════════════════════════════════════╗
║              Neural Compiler              ║
╚═══════════════════════════════════════════╝
""",
    "run": """
╔═══════════════════════════════════════════╗
║               Neural Runner               ║
╚═══════════════════════════════════════════╝
""",
    "debug": """
╔═══════════════════════════════════════════╗
║               Neural Debugger             ║
╚═══════════════════════════════════════════╝
""",
    "no-code": """
╔═══════════════════════════════════════════╗
║            Neural No-Code UI              ║
╚═══════════════════════════════════════════╝
"""
}

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Spinner animation for loading
class Spinner:
    def __init__(self, message="Processing", delay=0.1):
        self.spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.message = message
        self.delay = delay
        self.running = False
        self.spinner_thread = None

    def spin(self):
        i = 0
        while self.running:
            sys.stdout.write(f"\r{Colors.CYAN}{self.spinner_chars[i]}{Colors.ENDC} {self.message}")
            sys.stdout.flush()
            time.sleep(self.delay)
            i = (i + 1) % len(self.spinner_chars)
        sys.stdout.write("\r" + " " * (len(self.message) + 2) + "\r")
        sys.stdout.flush()

    def start(self):
        self.running = True
        self.spinner_thread = threading.Thread(target=self.spin)
        self.spinner_thread.daemon = True
        self.spinner_thread.start()

    def stop(self):
        self.running = False
        if self.spinner_thread:
            self.spinner_thread.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

# Progress bar
def progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{Colors.CYAN}{bar}{Colors.ENDC}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

# Neural network animation
def animate_neural_network(duration=3):
    """
    Animate a neural network with data flowing through it
    """
    layers = [
        ["○", "○", "○", "○"],  # Input layer
        ["○", "○", "○"],       # Hidden layer 1
        ["○", "○"],            # Hidden layer 2
        ["○"]                  # Output layer
    ]

    # Animation frames for data flow
    data_positions = []
    for i in range(len(layers) - 1):
        for _ in range(3):  # 3 frames per layer transition
            data_positions.append(i)

    start_time = time.time()
    frame = 0

    try:
        while time.time() - start_time < duration:
            sys.stdout.write("\r" + " " * 50 + "\r")  # Clear line

            # Determine which layer transition to highlight
            layer_idx = data_positions[frame % len(data_positions)]

            # Print the network with highlighted connections
            for i, layer in enumerate(layers):
                layer_str = " ".join(layer)
                if i == layer_idx:
                    sys.stdout.write(f"{layer_str} {Colors.CYAN}→{Colors.ENDC} ")
                elif i == layer_idx + 1:
                    sys.stdout.write(f"{Colors.CYAN}{layer_str}{Colors.ENDC} ")
                else:
                    sys.stdout.write(f"{layer_str} ")
                    if i < len(layers) - 1:
                        sys.stdout.write("→ ")

            sys.stdout.flush()
            time.sleep(0.2)
            frame += 1

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\n")

# Success and error messages
def print_success(message):
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")

def print_error(message):
    print(f"{Colors.RED}✗ {message}{Colors.ENDC}")

def print_warning(message):
    print(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")

def print_info(message):
    print(f"{Colors.BLUE}ℹ {message}{Colors.ENDC}")

# Print command header
def print_command_header(command):
    header = COMMAND_HEADERS.get(command, NEURAL_LOGO_SMALL)
    print(f"{Colors.CYAN}{header}{Colors.ENDC}")

# Print Neural logo with version
def print_neural_logo(version="1.0.0"):
    print(f"{Colors.CYAN}{NEURAL_LOGO}{Colors.ENDC}")
    print(f"{Colors.BOLD}Neural DSL {version}{Colors.ENDC}")
    print(f"{Colors.BLUE}A Neural Network Programming Language{Colors.ENDC}")
    print()

# Print help command with better formatting
def print_help_command(ctx, commands):
    """Print help command with better formatting.

    Args:
        ctx: Click context
        commands: List of commands
    """
    print(f"{Colors.CYAN}{NEURAL_LOGO_SMALL}{Colors.ENDC}")
    print(f"{Colors.BOLD}Neural CLI Help{Colors.ENDC}")
    print(f"{Colors.BLUE}A Neural Network Programming Language{Colors.ENDC}\n")

    print(f"{Colors.CYAN}Usage:{Colors.ENDC}")
    print(f"  neural [OPTIONS] COMMAND [ARGS]...\n")

    print(f"{Colors.CYAN}Options:{Colors.ENDC}")
    print(f"  -v, --verbose  Enable verbose logging")
    print(f"  --help         Show this message and exit.")
    print(f"  --version      Show the version and exit.\n")

    print(f"{Colors.CYAN}Commands:{Colors.ENDC}")
    for cmd_name in sorted(commands.keys()):
        cmd = commands[cmd_name]
        help_text = cmd.help or ""
        print(f"  {Colors.BOLD}{cmd_name.ljust(15)}{Colors.ENDC}{help_text}")

    print(f"\n{Colors.BLUE}Run 'neural COMMAND --help' for more information on a command.{Colors.ENDC}")

# Execute a function with a spinner
def with_spinner(func, message="Processing", *args, **kwargs):
    with Spinner(message) as spinner:
        result = func(*args, **kwargs)
    return result

# Main function to test the aesthetics
if __name__ == "__main__":
    print_neural_logo("1.0.0")
    print_command_header("visualize")

    print("Starting shape propagation...")
    for i in range(101):
        progress_bar(i, 100, prefix='Progress:', suffix='Complete', length=50)
        time.sleep(0.02)

    print_success("Shape propagation completed successfully!")

    with Spinner("Generating visualization"):
        time.sleep(3)

    print_success("Visualization generated successfully!")
    animate_neural_network(3)
