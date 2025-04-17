from PyDictionary import PyDictionary

dictionary = PyDictionary()

def get_word_info(word):
    try:
        meaning = dictionary.meaning(word)
        example = dictionary.get_examples(word)
        if meaning:
            meaning_text = list(meaning.values())[0][0]
        else:
            meaning_text = "Not found"
        return meaning_text, example
    except:
        return "Not found", []
