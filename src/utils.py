from pathlib import Path


def mkdir(din):
    """
    :param din directory to make
    """
    Path(din).absolute().mkdir(exist_ok=True, parents=True)

