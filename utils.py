import sys
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


def dir_checker(path: str):
    result = {}
    data_dir = pathlib.Path(f'{path}')
    for p_dir in data_dir.iterdir():
        counter = [1 for x in list(os.scandir(p_dir)) if x.is_file()]
        if len(counter) > 100:
            sep_dir = str(p_dir).split(sep='/')
            # print(p_dir, len(counter))
            result[sep_dir[1]] = len(counter)
    return result


def counter_helper():
    old = dir_checker(path='./old_data')
    new = dir_checker(path='./cells')

    merge = {}
    for k, v in new.items():
        if k in old:
            merge[k] = v + old[k]

    return merge


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


def sampler(data_path: str, ratio_train: float, case: str):
    """Сборщик сэмплов."""
    data_dir = pathlib.Path(f'{data_path}')
    merged_cells_info = counter_helper()
    cell_types = list(merged_cells_info.keys())
    min_size = min(list(dir_checker(path=data_path).values()))
    for name in cell_types:
        cells_paths = list(data_dir.glob(f'./{name}/*.bmp'))

        # Формирование тренировочной выборки
        path_to_train = f'./img_data/{case}/train/{name}/'
        os.makedirs(path_to_train, exist_ok=True)
        train_size = int(min_size * ratio_train)
        train_cells = random_fill_dir(
            ds_size=train_size,
            dir_name=path_to_train,
            paths=cells_paths,
        )

        # Формирование тестовой выборки
        path_to_test = f'./img_data/{case}/test/{name}/'
        os.makedirs(path_to_test, exist_ok=True)
        test_size = int(min_size * (1 - ratio_train))
        random_fill_dir(
            ds_size=test_size,
            dir_name=path_to_test,
            paths=cells_paths,
            other_check_set=train_cells,
        )


if __name__ == "__main__":
    # fetcher(key=sys.argv[1])
    # dir_checker()
    # counter_helper()
    sampler(
        data_path='./cells',
        ratio_train=0.8,
        case='new'
    )
