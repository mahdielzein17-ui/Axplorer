from src.envs.cycle import SquareEnvironment
from src.envs.isosceles import IsoscelesEnvironment
from src.envs.sphere import SphereEnvironment
from src.envs.ramsey import RamseyEnvironment
from src.envs.hexagon import HexagonEnvironment

ENVS = {
    "square": SquareEnvironment,
    "isosceles": IsoscelesEnvironment,
    "sphere": SphereEnvironment,
    "ramsey": RamseyEnvironment,
    "hexagon": HexagonEnvironment,
}


def build_env(params):
    """
    Build environment.
    """
    env = ENVS[params.env_name](params)
    return env
