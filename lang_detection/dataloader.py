import pandas as pd
import glob
import unidecode


def get_letter_freq(text: str):
    standard_gen_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    letter_freq = {letter: 0 for letter in standard_gen_letters}
    for letter in text:
        if letter in letter_freq:
            letter_freq[letter] += 1
        else:
            letter_freq[letter] = 1

    letters_in_list = list(map(lambda x: x[1], sorted(letter_freq.items(), key=lambda x: x[0])))

    return letters_in_list, letter_freq


def sanitize(text: str):
    text = unidecode.unidecode(text)
    text = text.upper()
    text = "".join(filter(str.isalpha, text))
    return text


def load_texts(pattern):
    lang_files = glob.glob(pattern)
    df = pd.DataFrame(columns=['lang', 'vector', 'debug_text'])

    for lang_file in lang_files:
        lang_name, _ = lang_file.split("\\")[-1].split(".")

        with open(lang_file, encoding="utf-8") as file:
            for unsafe_line in file:
                if len(unsafe_line) < 26 * 2:
                    continue
                line = sanitize(unsafe_line)

                freq_list, _ = get_letter_freq(line)
                df.loc[len(df)] = [lang_name, freq_list, unsafe_line]

    return df

