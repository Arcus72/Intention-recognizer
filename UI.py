import tkinter as tk
from PIL import Image, ImageTk

root = tk.Tk()
root.title("Program do sylwetek")
root.geometry("800x600")
root.resizable(False, False)

def wybrano_uruchom():
    print("Uruchom")

def wybrano_statystyka():
    print("Statystyka")

def wybrano_uczenie():
    print("Uczenie")

def wybrano_wyjdź():
    root.destroy()

# Zdjęcie na tło
img = Image.open("góry_IO.jpg").resize((800, 600), Image.LANCZOS)
bg = ImageTk.PhotoImage(img)
label = tk.Label(root, image=bg)
label.place(x=0, y=0)



kolumna = 1
wiersz = 1

button_uruchom = tk.Button(root, text="Uruchom", command=wybrano_uruchom).grid(row=wiersz, column=kolumna, sticky="NSEW")
button_statystyka = tk.Button(root, text="Statystyka", command=wybrano_statystyka).grid(row=wiersz+1, column=kolumna, sticky="NSEW")
button_uczenie = tk.Button(root, text="Uczenie", command=wybrano_uczenie).grid(row=wiersz+2, column=kolumna, sticky="NSEW")
button_wyjdź = tk.Button(root, text="Wyjdź", command=wybrano_wyjdź).grid(row=wiersz+3, column=kolumna, sticky="NSEW")

root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=2)
root.columnconfigure(2, weight=1)

root.rowconfigure(0, weight=5)
root.rowconfigure(1, weight=1)
root.rowconfigure(2, weight=1)
root.rowconfigure(3, weight=1)
root.rowconfigure(4, weight=1)
root.rowconfigure(5, weight=5)

root.mainloop()