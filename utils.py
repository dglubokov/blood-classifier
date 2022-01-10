import os
import pathlib
import shutil
import re
import random


def fetcher(key: str):
    # создание папки, в которую добавляются клетки
    os.makedirs(f'./cells/{key}', exist_ok=True)

    # итерация по папкам
    data_dir = pathlib.Path("./new_data")

    # открыть папку, найти и переместить файл с ключевым словом
    for p_dir in data_dir.iterdir():
        paths_images = list(p_dir.glob('./*.bmp'))
        for cell_path in paths_images:
            if re.search(key, str(cell_path)):
                print(str(cell_path))
                shutil.move(str(cell_path), f'./cells/{key}/')


def dir_checker(path: str, size: int):
    result = []
    data_dir = pathlib.Path(f'{path}')
    for p_dir in data_dir.iterdir():
        counter = [1 for x in list(os.scandir(p_dir)) if x.is_file()]
        if len(counter) > size:
            # print(p_dir, len(counter))
            result.append(str(p_dir))
    return result


def random_fill_dir(ds_size, dir_name, paths, other_check_set=set()):
    """Функция для заполнения папок случайными выборками."""
    check_set = set()
    while len(check_set) != ds_size:
        # Выбираем случайный номер из списка
        random_path_i = random.randrange(len(paths))
        p = paths[random_path_i]

        # Если не использовали путь, добавляем в выборку
        if p not in check_set and p not in other_check_set:
            shutil.copyfile(p, dir_name + p.name)
            check_set.add(p)
    return check_set


def resampler(path: str, size: int, save_to: str, image_format: str = '.bmp'):
    result = dir_checker(path, size)
    save_path = save_to + 'resampled/'
    os.makedirs(save_path, exist_ok=True)

    for c in result:
        class_name = c.split(sep='/')[-1] + '/'

        data_dir = pathlib.Path(path)
        cells_paths = list(data_dir.glob(f'{class_name}*{image_format}'))

        class_new_path = save_path + class_name
        os.makedirs(class_new_path, exist_ok=True)
        random_fill_dir(
            ds_size=size,
            dir_name=class_new_path,
            paths=cells_paths, 
        )


if __name__ == "__main__":
    resampler(
        path='./data/cells/',
        size=500,
        save_to='./data/'
    )
