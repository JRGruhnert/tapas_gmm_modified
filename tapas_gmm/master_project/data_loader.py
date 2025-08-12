from tapas_gmm.master_project.state import State, StateSpace
from tapas_gmm.master_project.task import Task, TaskSpace


class DataLoader:
    def __init__(self, state_space: StateSpace, task_space: TaskSpace):
        self._state_space = state_space
        self._task_space = task_space
        self._states: list[State] = self.load_states()
        self._tasks: list[Task] = self.load_tasks()

    def load_states(self):
        """Load states from the state space."""
        self._states = State.convert_to_states(self._state_space)

    def load_tasks(self):
        """Load tasks from the task space."""
        self._tasks = Task.convert_to_tasks(self._task_space)

    @property
    def states(self) -> list[State]:
        """Returns the loaded states."""
        return self._states

    @property
    def tasks(self) -> list[Task]:
        """Returns the loaded tasks."""
        return self._tasks
