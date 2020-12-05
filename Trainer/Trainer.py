from typing import Sequence, Union


class Trainer:
    def __init__(self, loss: str = "cross_entropy", tol: float = 0.01, pin_memory: bool = False, num_worker: int = 0,
                 save_path: str = "", bitch: Union[Sequence[int], int] = None):

        self.tol = tol
        self.pin_memory = pin_memory
        self.num_work = num_worker
        self.save_path = save_path
