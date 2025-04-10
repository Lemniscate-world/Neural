"""
Welcome message module for Neural CLI.
Displays a welcome message the first time the CLI is run.
"""

import os
import sys
from pathlib import Path
from .cli_aesthetics import Colors, print_neural_logo, animate_neural_network

WELCOME_MESSAGE = f"""
{Colors.CYAN}Welcome to Neural CLI!{Colors.ENDC}

Neural is a powerful tool for building, training, and visualizing neural networks.
Here are some commands to get you started:

  {Colors.BOLD}neural version{Colors.ENDC}              - Show version information
  {Colors.BOLD}neural visualize <file>{Colors.ENDC}     - Visualize a neural network
  {Colors.BOLD}neural compile <file>{Colors.ENDC}       - Compile a Neural DSL file
  {Colors.BOLD}neural run <file>{Colors.ENDC}           - Run a compiled model
  {Colors.BOLD}neural debug <file>{Colors.ENDC}         - Debug a neural network
  {Colors.BOLD}neural no-code{Colors.ENDC}              - Launch the no-code interface
  {Colors.BOLD}neural clean{Colors.ENDC}                - Clean generated files

For more information, run {Colors.BOLD}neural --help{Colors.ENDC} or {Colors.BOLD}neural <command> --help{Colors.ENDC}

{Colors.YELLOW}Happy neural network building!{Colors.ENDC}
"""

def show_welcome_message():
    """Show the welcome message if it's the first time the CLI is run.

    Returns:
        bool: True if the welcome message was shown, False otherwise.
    """
    # Get the user's home directory
    home_dir = Path.home()

    # Create the .neural directory if it doesn't exist
    neural_dir = home_dir / ".neural"
    neural_dir.mkdir(exist_ok=True)

    # Check if the welcome message has been shown before
    welcome_file = neural_dir / "welcome_shown"
    if not welcome_file.exists():
        # Show the welcome message
        print_neural_logo()
        print(WELCOME_MESSAGE)

        # Show a brief animation
        print("Here's a preview of what Neural can visualize:")
        animate_neural_network(3)

        # Create the welcome file to indicate that the welcome message has been shown
        welcome_file.touch()

        # Ask the user if they want to continue
        print(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")
        input()

        return True  # Welcome message was shown

    return False  # Welcome message was not shown

if __name__ == "__main__":
    show_welcome_message()
