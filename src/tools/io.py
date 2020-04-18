from pathlib import Path


def mkdir(din):
    """
    :param din directory to make
    """
    Path(din).absolute().mkdir(exist_ok=True, parents=True)


def unique_path(directory, name_pattern):
    counter = 0
    while True:
        counter += 1
        path = directory / name_pattern.format(counter)
        if not path.exists():
            return path


def tree(directory):
    print(f'+ {directory}')
    for path in sorted(directory.rglob('*')):
        depth = len(path.relative_to(directory).parts)
        spacer = '    ' * depth
        print(f'{spacer}+ {path.name}')


if __name__ == '__main__':
    # sample unique_path
    path = unique_path(Path.cwd(), 'test{:03d}.txt')
    print(tree(Path.cwd()))
