# Logging Configuration Files

This directory contains all logging-related YAML configuration files for the Decoder4OT project. These configurations are part of the Hydra configuration system and should be used with Hydra-based training scripts.

## Files Overview

### 1. `logging_config.yaml`
**Main logging configuration file** with comprehensive logging settings including:
- Basic logging configuration (level, format, handlers)
- Hydra-specific logging setup
- Job logging configuration
- Logging presets (quiet, verbose, production)
- Usage examples and documentation

### 2. `my_custom_config.yaml`
**Your custom training configuration** with added logging section:
- Complete training configuration
- Model, training, tokenizer, and data settings
- Logging configuration section

### 3. `hydra_config.yaml`
**Hydra main configuration** (copied from `decoder4ot/configs/hydra/config.yaml`):
- Hydra job logging setup
- Console and file handlers
- Default configuration structure

### 4. `global_config.yaml`
**Global configuration reference** (copied from `decoder4ot/configs/hydra/global_config.yaml`):
- Complete configuration overview
- All available options documented
- Usage examples and variants

### 5. `my_hydra_config.yaml`
**Your Hydra configuration** (copied from `decoder4ot/configs/hydra/my_config.yaml`):
- Your specific Hydra settings

## Usage

### Basic Logging Setup

To use the logging configuration in your training scripts:

```python
import yaml
import logging.config

# Load logging configuration
with open('decoder4ot/configs/hydra/logging/logging_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Apply logging configuration
logging.config.dictConfig(config['logging'])
```

### Different Logging Levels

You can modify the logging level in `logging_config.yaml`:

```yaml
logging:
  level: INFO  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Logging Presets

The `logging_config.yaml` includes predefined presets:

1. **Quiet Logging** (minimal output):
   ```yaml
   logging.level: WARNING
   logging.console.level: WARNING
   logging.file.enabled: false
   ```

2. **Verbose Logging** (detailed output):
   ```yaml
   logging.level: DEBUG
   logging.console.level: DEBUG
   logging.file.level: DEBUG
   ```

3. **Production Logging** (balanced):
   ```yaml
   logging.level: INFO
   logging.file.max_bytes: 10485760  # 10MB
   logging.file.backup_count: 10
   ```

### File Logging

Configure file logging in `logging_config.yaml`:

```yaml
logging:
  file:
    enabled: true
    filename: "training.log"
    mode: "w"
    level: INFO
    max_bytes: 10485760  # 10MB
    backup_count: 5
```

### Console Output

Configure console output:

```yaml
logging:
  console:
    enabled: true
    level: INFO
    stream: "ext://sys.stdout"
```

### Log Rotation

Enable log rotation to manage log file sizes:

```yaml
logging:
  rotation:
    enabled: true
    max_size: "10MB"
    backup_count: 5
    when: "midnight"
    interval: 1
```

## Integration with Hydra

If you're using Hydra for configuration management, you can include the logging configuration:

```yaml
# In your main config file
defaults:
  - _self_
  - logging: default  # Reference the logging config
  - model: default
  - training: default
```

## Custom Log Format

You can customize the log format:

```yaml
logging:
  format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  date_format: '%Y-%m-%d %H:%M:%S'
```

## Error Logging

Separate error logging is configured in the Hydra setup:

```yaml
hydra_logging:
  handlers:
    error_file:
      class: logging.FileHandler
      level: ERROR
      filename: ${hydra:runtime.output_dir}/errors.log
```

## Examples

### Using with Python Scripts

```python
import yaml
import logging

# Load configuration
with open('configs/logging/logging_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Setup logging
logging.config.dictConfig(config['logging'])

# Use logger
logger = logging.getLogger(__name__)
logger.info("Training started")
logger.debug("Detailed debug information")
logger.warning("Warning message")
logger.error("Error occurred")
```

### Using with Hydra

```bash
# Run with default logging
python -m decoder4ot.train_regression_singletarget_hydra

# Run with quiet logging
python -m decoder4ot.train_regression_singletarget_hydra logging=quiet

# Run with verbose logging
python -m decoder4ot.train_regression_singletarget_hydra logging=verbose
```

## File Structure

```
decoder4ot/configs/hydra/logging/
├── README.md                    # This file
├── logging_config.yaml          # Main logging configuration
├── my_custom_config.yaml        # Your custom config with logging
├── hydra_config.yaml           # Hydra main config
├── global_config.yaml          # Global config reference
└── my_hydra_config.yaml       # Your Hydra config
```

## Notes

- All log files are created in the output directory specified by Hydra
- Log rotation helps manage disk space
- Different logging levels can be set for console vs file output
- The configuration supports both standalone and Hydra-based usage
- Error logs are automatically separated from regular logs 