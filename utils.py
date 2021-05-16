from collections import Counter
from tqdm import tqdm
from jamo import hangul_to_jamo

from multiprocessing import Pool
from itertools import repeat


JAMO_LEADS = "".join([chr(_) for _ in range(0x1100, 0x1113)])
JAMO_VOWELS = "".join([chr(_) for _ in range(0x1161, 0x1176)])
JAMO_TAILS = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])

JAMO = JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS


def load_data(path):
    with open(path, 'r') as f:
        data = f.readlines()

    return [x.strip() for x in data]


def text_to_idx():
    pass


def index_to_text():
    pass


# for multiprocessing
def split_data(n, data):
    return [data[i:i + int(len(data) / n) + 1] for i in range(0, len(data), int(len(data) / n) + 1)]


# sentence to char
def to_jamo(data, use_counter=False):
    if use_counter:
        analysis_result = Counter()
    else:
        analysis_result = set()

    for x in tqdm(data):
        analysis_result.update(hangul_to_jamo(x))

    return analysis_result


def text_to_char(data, n=16, use_counter=False):
    splited_data = split_data(n, data)

    with Pool(n) as p:
        results = p.starmap(to_jamo, zip(splited_data, repeat(use_counter)))

    if use_counter:
        return sum(results, Counter())
    else:
        return set(set.union(*results))


if __name__ == '__main__':
    train_data = load_data('../train_data')

    counted_char_train_data = text_to_char(train_data, n=16, use_counter=True)

    counted_char_train_data.most_common()[160]
    symbol_list = [s[0] for s in counted_char_train_data.most_common()[:160]]

    print(symbol_list)

    symbol_set = set(symbol_list + list(JAMO))

    print(len(symbol_set))
    print(symbol_set)

