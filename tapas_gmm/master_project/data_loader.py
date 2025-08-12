from tapas_gmm.master_project.state import State, StateSpace
from tapas_gmm.master_project.task import Task, TaskSpace


class DataLoader:
    def __init__(self, state_space: StateSpace, task_space: TaskSpace):
        self._states: list[State] = State.from_json_list(state_space)
        self._tasks: list[Task] = Task.from_json_list(task_space)

    @property
    def states(self) -> list[State]:
        """Returns the loaded states."""
        return self._states

    @property
    def tasks(self) -> list[Task]:
        """Returns the loaded tasks."""
        return self._tasks
