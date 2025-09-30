"""
Configuration Manager for WebChasor
Handles loading and accessing configuration from YAML files
"""

import os
import yaml
import asyncio
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Centralized configuration manager for WebChasor system
    
    Loads configuration from YAML files and provides easy access to nested values.
    Supports environment-specific overrides and fallback values.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to config file. If None, uses default location.
        """
        self._config = {}
        self._config_path = config_path or self._get_default_config_path()
        self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        # Look for config file in project root
        project_root = Path(__file__).parent.parent
        config_file = project_root / "config" / "config.yaml"
        
        if config_file.exists():
            return str(config_file)
        
        # Fallback to src directory
        fallback_config = Path(__file__).parent / "config.yaml"
        if fallback_config.exists():
            return str(fallback_config)
        
        raise FileNotFoundError(f"Configuration file not found. Looked in: {config_file}, {fallback_config}")
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
            
            logger.info(f"Configuration loaded from: {self._config_path}")
            
            # Apply environment-specific overrides
            self._apply_environment_overrides()
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self._config_path}")
            self._config = self._get_default_config()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            self._config = self._get_default_config()
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}")
            self._config = self._get_default_config()
    
    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides"""
        env = self.get('system.environment', 'development')
        
        # Environment-specific overrides
        if env == 'production':
            self._config.setdefault('logging', {})['level'] = 'WARNING'
            self._config.setdefault('system', {})['debug'] = False
        elif env == 'testing':
            self._config.setdefault('logging', {})['level'] = 'DEBUG'
            self._config.setdefault('performance', {}).setdefault('timeouts', {})['action_execution'] = 30
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get minimal default configuration if file loading fails"""
        return {
            'system': {
                'name': 'WebChasor',
                'version': '1.0.0',
                'environment': 'development',
                'debug': True,
                'max_execution_rounds': 2
            },
            'models': {
                'synthesizer': {
                    'model_name': 'gpt-4',
                    'temperature': 0.1,
                    'max_tokens': 2000
                }
            },
            'ir_rag': {
                'web_scraping': {'enabled': False},
                'search': {'max_results': 10, 'location': 'Hong Kong', 'language': 'zh-cn'},
                'content': {'chunk_size': 500, 'max_passages_per_task': 3}
            },
            'logging': {
                'level': 'INFO',
                'decisions': {'enabled': True}
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key in dot notation (e.g., 'ir_rag.web_scraping.enabled')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Examples:
            config.get('ir_rag.web_scraping.enabled')
            config.get('models.synthesizer.temperature', 0.1)
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to parent dict
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section
        
        Args:
            section: Section name (e.g., 'ir_rag', 'models')
            
        Returns:
            Dictionary containing the section configuration
        """
        return self.get(section, {})
    
    def reload(self):
        """Reload configuration from file"""
        self._load_config()
    
    def save(self, path: Optional[str] = None):
        """
        Save current configuration to file
        
        Args:
            path: Path to save to. If None, uses current config path.
        """
        save_path = path or self._config_path
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to: {save_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def validate(self) -> bool:
        """
        Validate configuration completeness
        
        Returns:
            True if configuration is valid, False otherwise
        """
        required_sections = ['system', 'models', 'ir_rag', 'reasoning', 'productivity']
        
        for section in required_sections:
            if section not in self._config:
                logger.warning(f"Missing required configuration section: {section}")
                return False
        
        # Validate critical settings
        if not self.get('models.synthesizer.model_name'):
            logger.warning("Missing synthesizer model name")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """
        Get model configuration for specific model type
        
        Args:
            model_type: Type of model (router, planner, synthesizer, extractor)
            
        Returns:
            Model configuration dictionary
        """
        return self.get_section(f'models.{model_type}')
    
    def get_action_config(self, action_name: str) -> Dict[str, Any]:
        """
        Get action-specific configuration
        
        Args:
            action_name: Name of action (ir_rag, reasoning, productivity)
            
        Returns:
            Action configuration dictionary
        """
        return self.get_section(action_name.lower())
    
    def is_debug_enabled(self) -> bool:
        """Check if debug mode is enabled"""
        return self.get('system.debug', False)
    
    def is_decision_logging_enabled(self, component: str = None) -> bool:
        """
        Check if decision logging is enabled for a component
        
        Args:
            component: Component name (router, registry, etc.). If None, checks global setting.
            
        Returns:
            True if decision logging is enabled
        """
        if component:
            return self.get(f'logging.decisions.{component}', True)
        return self.get('logging.decisions.enabled', True)
    
    def get_timeout(self, timeout_type: str) -> int:
        """
        Get timeout value for specific operation
        
        Args:
            timeout_type: Type of timeout (action_execution, web_request, llm_request)
            
        Returns:
            Timeout value in seconds
        """
        return self.get(f'performance.timeouts.{timeout_type}', 30)
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"ConfigManager(path={self._config_path}, sections={list(self._config.keys())})"
    
    def __repr__(self) -> str:
        return self.__str__()


# Global configuration instance
_config_instance = None

# Global semaphore instances
_global_gate = None
_llm_gate = None

def get_config() -> ConfigManager:
    """
    Get global configuration instance (singleton pattern)
    
    Returns:
        ConfigManager instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    return _config_instance

def get_global_gate() -> asyncio.Semaphore:
    """
    Get global concurrent request semaphore (singleton pattern)
    
    Returns:
        Semaphore for limiting global concurrent requests
    """
    global _global_gate
    if _global_gate is None:
        max_requests = get_config().get('performance.concurrency.max_concurrent_requests', 100)
        _global_gate = asyncio.Semaphore(max_requests)
        logger.info(f"Global gate initialized with limit: {max_requests}")
    return _global_gate

def get_llm_gate() -> asyncio.Semaphore:
    """
    Get LLM concurrent request semaphore (singleton pattern)
    
    Returns:
        Semaphore for limiting LLM concurrent requests
    """
    global _llm_gate
    if _llm_gate is None:
        llm_limit = get_config().get('performance.concurrency.llm_concurrent_limit', 20)
        _llm_gate = asyncio.Semaphore(llm_limit)
        logger.info(f"LLM gate initialized with limit: {llm_limit}")
    return _llm_gate

def reload_config():
    """Reload global configuration"""
    global _config_instance
    if _config_instance:
        _config_instance.reload()
    else:
        _config_instance = ConfigManager()

# Convenience functions for common configuration access
def get_model_name(model_type: str) -> str:
    """Get model name for specific model type"""
    return get_config().get(f'models.{model_type}.model_name', 'gpt-4')

def get_web_scraping_enabled() -> bool:
    """Check if web scraping is enabled"""
    return get_config().get('ir_rag.web_scraping.enabled', False)

def get_max_search_results() -> int:
    """Get maximum number of search results"""
    return get_config().get('ir_rag.search.max_results', 10)

def get_chunk_size() -> int:
    """Get content chunk size"""
    return get_config().get('ir_rag.content.chunk_size', 500)

def is_debug_mode() -> bool:
    """Check if debug mode is enabled"""
    return get_config().is_debug_enabled()

# Example usage and testing
if __name__ == "__main__":
    # Test configuration manager
    config = ConfigManager()
    
    print("Testing Configuration Manager")
    print("=" * 50)
    
    # Test basic access
    print(f"System name: {config.get('system.name')}")
    print(f"Web scraping enabled: {config.get('ir_rag.web_scraping.enabled')}")
    print(f"Synthesizer model: {config.get('models.synthesizer.model_name')}")
    
    # Test section access
    ir_config = config.get_section('ir_rag')
    print(f"IR_RAG config keys: {list(ir_config.keys())}")
    
    # Test convenience functions
    print(f"Debug mode: {is_debug_mode()}")
    print(f"Web scraping: {get_web_scraping_enabled()}")
    print(f"Max search results: {get_max_search_results()}")
    
    # Test validation
    is_valid = config.validate()
    print(f"Configuration valid: {is_valid}")
    
    print("Configuration manager test completed") 