# JSON-Based Configuration System

This directory contains a refactored configuration system that replaces the enum-based approach with a cleaner JSON-based configuration.

## Files

- `config.json` - Main configuration file containing all state and task definitions
- `definitions_json.py` - Python module that loads and manages the JSON configuration
- `config_schema.json` - JSON schema for validating the configuration file
- `validate_config.py` - Script to validate the configuration against the schema
- `test_json_config.py` - Test script demonstrating the new system
- `migration_example.py` - Example showing how to migrate from the old enum-based system

## Advantages of JSON-Based Configuration

1. **Separation of Concerns**: Configuration data is separated from Python code
2. **Easy Modifications**: Changes can be made without editing Python files
3. **Better Maintainability**: Clear structure and external validation
4. **Version Control Friendly**: JSON changes are easier to track and review
5. **Schema Validation**: Automatic validation ensures configuration correctness
6. **Runtime Flexibility**: Configuration can be swapped without code changes

## Usage

### Basic Usage

```python
from definitions_json import get_config_manager, StateSpace, TaskSpace

# Get the config manager
config = get_config_manager()

# Get state information
ee_state = config.get_state("EE_State")
print(f"Identifier: {ee_state.identifier}")
print(f"Min: {ee_state.min}, Max: {ee_state.max}")

# Get task information
task = config.get_task("DrawerDoClose")
print(f"Preconditions: {task.precondition}")
print(f"Reversed: {task.reversed}")

# Get states by space
small_states = config.convert_to_states(StateSpace.SMALL)
all_tasks = config.convert_to_tasks(TaskSpace.ALL)
```

### Migration from Old System

**Old Way:**

```python
from definitions import State, Task

# Access state
ee_state_info = State.EE_State.value
task_info = Task.DrawerDoClose.value
```

**New Way:**

```python
from definitions_json import get_config_manager

config = get_config_manager()
ee_state_info = config.get_state("EE_State")
task_info = config.get_task("DrawerDoClose")
```

## Configuration Structure

### States

Each state has the following properties:

- `identifier`: Unique string identifier
- `type`: One of "Transform", "Quaternion", or "Scalar"
- `space`: One of "SMALL", "ALL", or "UNUSED"
- `success`: One of "AREA", "PRECISE", or "IGNORE"
- `min`: Minimum value(s) (number or array)
- `max`: Maximum value(s) (number or array)

### Tasks

Each task has the following properties:

- `precondition`: Dictionary mapping state names to values ("min", "max", or specific values)
- `space`: One of "SMALL" or "ALL"
- `reversed`: Boolean indicating if the task is reversed
- `ee_tp_start`: 7D pose array for end effector start position
- `obj_start`: 7D pose array for object start position
- `ee_hrl_start`: 7D pose array for end effector HRL start position

## Validation

To validate your configuration:

```bash
python validate_config.py
```

This will check that your `config.json` file conforms to the schema and provide statistics about your configuration.

## Testing

Run the test script to verify everything works:

```bash
python test_json_config.py
```

## Custom Configuration Paths

You can specify a custom configuration file path:

```python
from definitions_json import ConfigManager

config = ConfigManager("/path/to/your/config.json")
```

## Schema Validation

The configuration is automatically validated against `config_schema.json`. This ensures:

- All required fields are present
- Data types are correct
- Enum values are valid
- Array dimensions are correct (e.g., 7D poses)

## Best Practices

1. **Always validate** your configuration after making changes
2. **Use the schema** to understand the expected structure
3. **Test your changes** with the test script
4. **Keep backups** of working configurations
5. **Use version control** to track configuration changes
