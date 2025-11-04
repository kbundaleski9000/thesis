from benchmarl.environments import PettingZooTask, Task
from benchmarl.environments.common import TaskConfig
from benchmarl.utils import get_logger
from GridWorldAEC import GridWorldAECEnv # Assuming your environment is in a separate file

log = get_logger(__name__)

@TaskConfig
class GridWorldTask(PettingZooTask):
    @staticmethod
    def is_agent_group_homogeneous() -> bool:
        return False

    @staticmethod
    def num_agents() -> int:
        return 0

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Instantiate your custom environment
        self.env = GridWorldAECEnv(**kwargs)

    def make_env(self, **kwargs) -> GridWorldAECEnv:
        return self.env