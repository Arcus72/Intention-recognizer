from Recognizer import Recognizer
if __name__ == '__main__':

    print("1. Uruchom rozpoznawanie intencji")
    print("2. Dodaj sylwetkę")
    print("3. Statystyki")
    choice = input("Wybierz jedną z poniższych opcji: ")

    if choice == "1":
        recognizer = Recognizer()
        recognizer.activate()