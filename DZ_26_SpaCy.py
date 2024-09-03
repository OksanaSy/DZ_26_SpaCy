import spacy
from spacy.language import Language

#кастомний компонент
@Language.component("check_for_python")
def check_for_python(doc):
    doc.set_extension("has_python", default=False, force=True)
    if "Python" in doc.text:
        doc._.has_python = True

    return doc

nlp = spacy.load("en_core_web_md")

#кастомний компонент до пайплайну
nlp.add_pipe("check_for_python", last=True)

#перевірка
doc = nlp("I love programming in Python.")
print("Does the text contain 'Python'?", doc._.has_python)  # Output: True

doc2 = nlp("I love programming in Java.")
print("Does the text contain 'Python'?", doc2._.has_python)  # Output: False

#результати
#Does the text contain 'Python'? True
#Does the text contain 'Python'? False

