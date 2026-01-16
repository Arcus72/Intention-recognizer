import tkinter as tk
from collections import Counter

from PIL import Image, ImageTk
from tkinter import messagebox

from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils.VideoPoseStream import VideoPoseStream
from utils.Skeleton_generator import Skeleton_generator
from utils.recognize_intention import recognize_intention
from utils.train_models import train_models
import os


root = tk.Tk()
root.title("Program do sylwetek")
root.geometry("800x600")
root.resizable(False, False)

def start_handler():
    skeleton_generator = Skeleton_generator(False)
    videoPoseStream = VideoPoseStream(skeleton_generator.generate, recognize_intention)
    videoPoseStream.activate()

def count_photos(folder="dataset/pictures"):
    counter = Counter()

    if not os.path.exists(folder):
        return counter

    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            label = file.split("_")[0]
            counter[label] += 1

    return counter


def statistic_handler():
    for widget in root.winfo_children():
        if isinstance(widget, tk.Frame):
            widget.destroy()

    dane = count_photos()

    if not dane:
        print("Brak zdjęć w folderze dataset/pictures")
        return

    labels = list(dane.keys())
    sizes = list(dane.values())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title("Rozkład zdjęć w zbiorze")
    ax.axis("equal")

    frame = tk.Frame(root, bg="white")
    frame.place(relx=0.5, rely=0.5, anchor="center")

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()


def learning_handler():
    pictures_location = "dataset\pictures"
    data_outcome_folder = "dataset\processed_data"

    messagebox.showinfo("info", "Przetwarzanie zdjęć")
    skeleton_generator = Skeleton_generator(True)
    skeleton_generator.process_photos(pictures_location, data_outcome_folder)

    messagebox.showinfo("info", "Uczenie maszyny")
    train_models()


def exit_handler():
    root.destroy()


if __name__ == "__main__":
    img = Image.open("assets/background.png").resize((800, 600), Image.LANCZOS)
    bg = ImageTk.PhotoImage(img)
    label = tk.Label(root, image=bg)
    label.place(x=0, y=0)

    column = 1
    row = 1

    tk.Button(root, text="Uruchom", command=start_handler).grid(row=row, column=column, sticky="NSEW", pady=(0, 10))
    tk.Button(root, text="Statystyka", command=statistic_handler).grid(row=row+1, column=column, sticky="NSEW", pady=(0, 10))
    tk.Button(root, text="Uczenie", command=learning_handler).grid(row=row+2, column=column, sticky="NSEW", pady=(0, 10))
    tk.Button(root, text="Wyjdź", command=exit_handler).grid(row=row+3, column=column, sticky="NSEW", pady=(0, 10))

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