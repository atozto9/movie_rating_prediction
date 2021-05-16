from collections import Counter
from tqdm import tqdm
from jamo import hangul_to_jamo

from multiprocessing import Pool
from itertools import repeat

import utils

JAMO_LEADS = "".join([chr(_) for _ in range(0x1100, 0x1113)])
JAMO_VOWELS = "".join([chr(_) for _ in range(0x1161, 0x1176)])
JAMO_TAILS = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])

JAMO = JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS


class KoreanText:
    def __init__(self, symbol_from_data=False, most_n=160):
        if symbol_from_data:
            text_data = utils.load_data('../train_data')
            phonemes = text_to_char(text_data, n=16, use_counter=True)
            self.phonemes_list = [s[0] for s in phonemes.most_common()[:most_n]]

            self.phonemes_list = sorted(list(set(self.phonemes_list + list(JAMO))))

            self.symbol_dic = {x: i for i, x in enumerate(self.phonemes_list)}
        else:
            self.phonemes_list = [' ', '!', '%', '&', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'S', 'T', 'V', 'X', '[', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '~', 'ᄀ', 'ᄁ', 'ᄂ', 'ᄃ', 'ᄄ', 'ᄅ', 'ᄆ', 'ᄇ', 'ᄈ', 'ᄉ', 'ᄊ', 'ᄋ', 'ᄌ', 'ᄍ', 'ᄎ', 'ᄏ', 'ᄐ', 'ᄑ', 'ᄒ', 'ᅡ', 'ᅢ', 'ᅣ', 'ᅤ', 'ᅥ', 'ᅦ', 'ᅧ', 'ᅨ', 'ᅩ', 'ᅪ', 'ᅫ', 'ᅬ', 'ᅭ', 'ᅮ', 'ᅯ', 'ᅰ', 'ᅱ', 'ᅲ', 'ᅳ', 'ᅴ', 'ᅵ', 'ᆨ', 'ᆩ', 'ᆪ', 'ᆫ', 'ᆬ', 'ᆭ', 'ᆮ', 'ᆯ', 'ᆰ', 'ᆱ', 'ᆲ', 'ᆳ', 'ᆴ', 'ᆵ', 'ᆶ', 'ᆷ', 'ᆸ', 'ᆹ', 'ᆺ', 'ᆻ', 'ᆼ', 'ᆽ', 'ᆾ', 'ᆿ', 'ᇀ', 'ᇁ', 'ᇂ', '…', '★', '♡', '♥', 'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅋ', 'ㅎ', 'ㅏ', 'ㅗ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
            self.symbol_dic = {x: i for i, x in enumerate(self.phonemes_list)}

    def text_to_idx(self, text):
        return [self.symbol_dic[token] for token in hangul_to_jamo(text) if token in self.symbol_dic]

    def index_to_text(self, index):
        return [self.phonemes_list[i] for i in index]


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
    korean_text = KoreanText(symbol_from_data=True)

    print(korean_text.text_to_idx("안녕"))
    print(korean_text.index_to_text(korean_text.text_to_idx("안녕")))

    korean_text_ = KoreanText(symbol_from_data=False)

    print(sorted(korean_text.phonemes_list) == sorted(korean_text_.phonemes_list))
    print(sorted(korean_text.phonemes_list))
    print(sorted(korean_text_.phonemes_list))