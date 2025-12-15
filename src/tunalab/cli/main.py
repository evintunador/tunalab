import sys
import importlib
import pkgutil
from pathlib import Path
from typing import Dict, Type

from tunalab.cli.base import Command


def discover_commands() -> Dict[str, Type[Command]]:
    commands = {}
    
    import tunalab.cli.commands as commands_package
    commands_path = Path(commands_package.__file__).parent
    
    for _, module_name, _ in pkgutil.iter_modules([str(commands_path)]):
        if module_name.startswith('_'):
            continue
            
        module = importlib.import_module(f'tunalab.cli.commands.{module_name}')
        
        if hasattr(module, 'Command'):
            cmd_class = module.Command
            commands[cmd_class.name] = cmd_class
    
    return commands


def print_usage(commands: Dict[str, Type[Command]]):
    print("Usage: tunalab <command> [args...]")
    print("\nAvailable commands:")
    
    for name in sorted(commands.keys()):
        cmd = commands[name]
        print(f"  {name:15} {cmd.description}")


def main():
    commands = discover_commands()
    
    if len(sys.argv) < 2:
        print_usage(commands)
        sys.exit(1)
    
    command_name = sys.argv[1]
    
    if command_name in ['-h', '--help', 'help']:
        print_usage(commands)
        sys.exit(0)
    
    if command_name not in commands:
        print(f"Error: Unknown command '{command_name}'")
        print()
        print_usage(commands)
        sys.exit(1)
    
    # Remove the command name from argv so the command sees clean args
    sys.argv.pop(1)
    
    command_class = commands[command_name]
    command_class.main()
