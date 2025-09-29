#!/usr/bin/env python3
"""
WebChasor Configuration Management Tool
Simple CLI tool to view and modify configuration settings
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config_manager import get_config, reload_config

def show_config():
    """Display current configuration"""
    config = get_config()
    
    print("WebChasor Configuration")
    print("=" * 50)
    
    # System info
    print(f"System: {config.get('system.name')} v{config.get('system.version')}")
    print(f"Environment: {config.get('system.environment')}")
    print(f"Debug Mode: {config.get('system.debug')}")
    print()
    
    # Models
    print("Models:")
    for model_type in ['router', 'planner', 'synthesizer', 'extractor']:
        model_name = config.get(f'models.{model_type}.model_name', 'N/A')
        temp = config.get(f'models.{model_type}.temperature', 'N/A')
        print(f"  - {model_type.capitalize()}: {model_name} (temp: {temp})")
    print()
    
    # IR_RAG settings
    print("Information Retrieval:")
    print(f"  - Web Scraping: {'ON' if config.get('ir_rag.web_scraping.enabled') else 'OFF'}")
    print(f"  - Max Search Results: {config.get('ir_rag.search.max_results')}")
    print(f"  - Max Pages to Visit: {config.get('ir_rag.web_scraping.max_pages')}")
    print(f"  - Search Location: {config.get('ir_rag.search.location')}")
    print(f"  - Search Language: {config.get('ir_rag.search.language')}")
    print()
    
    # Logging
    print("Logging:")
    print(f"  - Level: {config.get('logging.level')}")
    print(f"  - Decision Logging: {'ON' if config.get('logging.decisions.enabled') else 'OFF'}")
    print()
    
    # Performance
    print("Performance:")
    print(f"  - Action Timeout: {config.get('performance.timeouts.action_execution')}s")
    print(f"  - Web Request Timeout: {config.get('performance.timeouts.web_request')}s")
    print(f"  - LLM Request Timeout: {config.get('performance.timeouts.llm_request')}s")

def toggle_web_scraping():
    """Toggle web scraping on/off"""
    config = get_config()
    current = config.get('ir_rag.web_scraping.enabled', False)
    new_value = not current
    
    config.set('ir_rag.web_scraping.enabled', new_value)
    config.save()
    
    status = "ON" if new_value else "OFF"
    print(f"Web scraping toggled: {status}")
    print("Configuration saved!")

def set_debug_mode(enabled: bool):
    """Enable/disable debug mode"""
    config = get_config()
    config.set('system.debug', enabled)
    config.save()
    
    status = "ON" if enabled else "OFF"
    print(f"Debug mode: {status}")
    print("Configuration saved!")

def set_log_level(level: str):
    """Set logging level"""
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
    if level.upper() not in valid_levels:
        print(f"Invalid log level. Must be one of: {', '.join(valid_levels)}")
        return
    
    config = get_config()
    config.set('logging.level', level.upper())
    config.save()
    
    print(f"Log level set to: {level.upper()}")
    print("Configuration saved!")

def set_model(model_type: str, model_name: str):
    """Set model for specific component"""
    valid_types = ['router', 'planner', 'synthesizer', 'extractor']
    if model_type not in valid_types:
        print(f"Invalid model type. Must be one of: {', '.join(valid_types)}")
        return
    
    config = get_config()
    config.set(f'models.{model_type}.model_name', model_name)
    config.save()
    
    print(f"{model_type.capitalize()} model set to: {model_name}")
    print("Configuration saved!")

def reset_config():
    """Reset configuration to defaults"""
    config_path = Path("config/config.yaml")
    backup_path = Path("config/config.yaml.backup")
    
    if config_path.exists():
        # Create backup
        import shutil
        shutil.copy2(config_path, backup_path)
        print(f"Backup created: {backup_path}")
    
    # Reload default config
    reload_config()
    print("Configuration reset to defaults")
    print("Done!")

def validate_config():
    """Validate current configuration"""
    config = get_config()
    is_valid = config.validate()
    
    if is_valid:
        print("Configuration is valid!")
    else:
        print("Configuration has issues. Check the logs for details.")
    
    return is_valid

def main():
    parser = argparse.ArgumentParser(description="WebChasor Configuration Tool")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Show config
    subparsers.add_parser('show', help='Display current configuration')
    
    # Toggle web scraping
    subparsers.add_parser('toggle-web', help='Toggle web scraping on/off')
    
    # Debug mode
    debug_parser = subparsers.add_parser('debug', help='Set debug mode')
    debug_parser.add_argument('state', choices=['on', 'off'], help='Enable or disable debug mode')
    
    # Log level
    log_parser = subparsers.add_parser('log-level', help='Set logging level')
    log_parser.add_argument('level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Log level')
    
    # Set model
    model_parser = subparsers.add_parser('set-model', help='Set model for component')
    model_parser.add_argument('type', choices=['router', 'planner', 'synthesizer', 'extractor'], help='Component type')
    model_parser.add_argument('model', help='Model name (e.g., gpt-4, gpt-3.5-turbo)')
    
    # Validate
    subparsers.add_parser('validate', help='Validate configuration')
    
    # Reset
    subparsers.add_parser('reset', help='Reset configuration to defaults')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'show':
            show_config()
        elif args.command == 'toggle-web':
            toggle_web_scraping()
        elif args.command == 'debug':
            set_debug_mode(args.state == 'on')
        elif args.command == 'log-level':
            set_log_level(args.level)
        elif args.command == 'set-model':
            set_model(args.type, args.model)
        elif args.command == 'validate':
            validate_config()
        elif args.command == 'reset':
            confirm = input("WARNING: This will reset all configuration to defaults. Continue? (y/N): ")
            if confirm.lower() == 'y':
                reset_config()
            else:
                print("Reset cancelled")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 