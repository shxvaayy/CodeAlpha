from googletrans import Translator

def translate_text(text, target_language='hi'):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text

# Example usage
while True:
    language=str(input("Type first two letters of language you want to translate in: "))
    text_to_translate = str(input(""))
    translated_text = translate_text(text_to_translate, language)
    print(f'Translated Text: {translated_text}')
