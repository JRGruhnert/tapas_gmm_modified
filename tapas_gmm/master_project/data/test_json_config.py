#!/usr/bin/env python3
"""
Migration script to test the JSON-based configuration system.
This script demonstrates how to use the new ConfigManager instead of the old enum-based approach.
"""

from tapas_gmm.master_project.data.definitions_json import (
    ConfigManager,
    StateSpace,
    TaskSpace,
)


def test_json_config():
    """Test the JSON-based configuration system"""

    # Initialize the config manager
    config = ConfigManager()

    print("=== Testing State Configuration ===")

    # Test getting a specific state
    ee_state = config.get_state("EE_State")
    print(
        f"EE_State: {ee_state.identifier}, type: {ee_state.type.value}, min: {ee_state.min}, max: {ee_state.max}"
    )

    drawer_transform = config.get_state("Drawer_Transform")
    print(
        f"Drawer_Transform: {drawer_transform.identifier}, space: {drawer_transform.space}"
    )

    # Test getting states by space
    small_states = config.convert_to_states(StateSpace.SMALL)
    print(
        f"SMALL space states ({len(small_states)}): {small_states[:5]}..."
    )  # Show first 5

    all_states = config.convert_to_states(StateSpace.ALL)
    print(f"ALL space states ({len(all_states)}): {len(all_states)} total states")

    print("\n=== Testing Task Configuration ===")

    # Test getting a specific task
    drawer_close = config.get_task("DrawerDoClose")
    print(f"DrawerDoClose preconditions: {drawer_close.precondition}")
    print(f"DrawerDoClose reversed: {drawer_close.reversed}")
    print(f"DrawerDoClose ee_hrl_start: {drawer_close.ee_hrl_start}")

    # Test getting tasks by space
    small_tasks = config.convert_to_tasks(TaskSpace.SMALL)
    print(f"SMALL space tasks ({len(small_tasks)}): {small_tasks[:5]}...")

    all_tasks = config.convert_to_tasks(TaskSpace.ALL)
    print(f"ALL space tasks ({len(all_tasks)}): {len(all_tasks)} total tasks")

    # Test finding state by identifier
    try:
        state_name = config.from_string("ee_euler_something")
        print(f"Found state for 'ee_euler': {state_name}")
    except NotImplementedError as e:
        print(f"State lookup error: {e}")

    print("\n=== Comparison with Original System ===")
    print(
        "The new JSON-based system provides the same functionality but with several advantages:"
    )
    print("1. Configuration is externalized to JSON files")
    print("2. Easier to modify without changing Python code")
    print("3. Better separation of data and logic")
    print("4. More maintainable and readable")
    print("5. Can be easily validated against JSON schemas")


if __name__ == "__main__":
    test_json_config()
