"""
Example showing how to migrate from the old enum-based system to the new JSON-based system.
"""

# OLD WAY (using enums):
# from definitions import State, Task, StateSpace, TaskSpace, convert_to_states, convert_to_tasks
#
# # Get state info
# ee_state_info = State.EE_State.value
# print(f"EE State identifier: {ee_state_info.identifier}")
#
# # Get task info
# drawer_task_info = Task.DrawerDoClose.value
# print(f"Drawer task preconditions: {drawer_task_info.precondition}")
#
# # Get states by space
# small_states = convert_to_states(StateSpace.SMALL)


# NEW WAY (using JSON config):
from definitions_json import get_config_manager, StateSpace, TaskSpace

# Get the config manager
config = get_config_manager()

# Get state info
ee_state_info = config.get_state("EE_State")
print(f"EE State identifier: {ee_state_info.identifier}")

# Get task info
drawer_task_info = config.get_task("DrawerDoClose")
print(f"Drawer task preconditions: {drawer_task_info.precondition}")

# Get states by space
small_states = config.convert_to_states(StateSpace.SMALL)


# Example of how you might update a function that used the old system:
def example_function_old():
    """Old way of using the configuration"""
    # This would have used:
    # from definitions import State, Task
    # states = [State.EE_Transform, State.EE_State, ...]
    # task_info = Task.DrawerDoClose.value
    pass


def example_function_new():
    """New way using JSON configuration"""
    config = get_config_manager()

    # Get specific states
    ee_transform = config.get_state("EE_Transform")
    ee_state = config.get_state("EE_State")

    # Get task info
    task_info = config.get_task("DrawerDoClose")

    # Use the information the same way as before
    print(f"Task preconditions: {task_info.precondition}")
    print(f"State identifier: {ee_transform.identifier}")


if __name__ == "__main__":
    example_function_new()
