#!/usr/bin/env python3
"""
Validation script for the JSON configuration.
"""

import json
import jsonschema
from pathlib import Path


def validate_config():
    """Validate the configuration JSON against the schema"""

    # Load the schema
    schema_path = Path(__file__).parent / "config_schema.json"
    with open(schema_path, "r") as f:
        schema = json.load(f)

    # Load the config
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    try:
        # Validate the config against the schema
        jsonschema.validate(config, schema)
        print("✅ Configuration is valid!")

        # Print some statistics
        print(f"📊 Configuration statistics:")
        print(f"   - States: {len(config['states'])}")
        print(f"   - Tasks: {len(config['tasks'])}")

        # Count by categories
        state_types = {}
        state_spaces = {}
        for state_name, state_data in config["states"].items():
            state_type = state_data["type"]
            state_space = state_data["space"]
            state_types[state_type] = state_types.get(state_type, 0) + 1
            state_spaces[state_space] = state_spaces.get(state_space, 0) + 1

        print(f"   - State types: {dict(state_types)}")
        print(f"   - State spaces: {dict(state_spaces)}")

        task_spaces = {}
        reversed_tasks = 0
        for task_name, task_data in config["tasks"].items():
            task_space = task_data["space"]
            task_spaces[task_space] = task_spaces.get(task_space, 0) + 1
            if task_data["reversed"]:
                reversed_tasks += 1

        print(f"   - Task spaces: {dict(task_spaces)}")
        print(f"   - Reversed tasks: {reversed_tasks}")

        return True

    except jsonschema.exceptions.ValidationError as e:
        print(f"❌ Configuration validation failed:")
        print(f"   Error: {e.message}")
        print(f"   Path: {' -> '.join(str(p) for p in e.absolute_path)}")
        return False

    except Exception as e:
        print(f"❌ Unexpected error during validation: {e}")
        return False


if __name__ == "__main__":
    validate_config()
