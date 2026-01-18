import os
import tkinter as tk
from tkinter import messagebox, ttk
from collections import Counter
import cv2
from PIL import Image, ImageTk
from pygrabber.dshow_graph import FilterGraph
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils.VideoPoseStream import VideoPoseStream
from utils.Skeleton_generator import Skeleton_generator
from utils.recognize_intention import recognize_intention
from utils.train_models import train_models


class SilhouetteApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Program do sylwetek")
        self.root.geometry("800x600")
        self.root.resizable(False, False)

        self.cameras_details = self._get_detailed_camera_list()

        self._setup_layout()
        self._create_widgets()

    def _get_detailed_camera_list(self):
        active_cameras = []
        try:
            graph = FilterGraph()
            device_names = graph.get_input_devices()
            for index, name in enumerate(device_names):
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        active_cameras.append({"id": index, "name": name})
                    cap.release()
            return active_cameras if active_cameras else [{"id": 0, "name": "Brak kamer"}]
        except Exception:
            return [{"id": 0, "name": "Błąd sterownika"}]

    def _setup_layout(self):
        try:
            img = Image.open("assets/background.png").resize((800, 600), Image.LANCZOS)
            self.bg_photo = ImageTk.PhotoImage(img)
            tk.Label(self.root, image=self.bg_photo).place(x=0, y=0)
        except:
            print("Nie znaleziono pliku tła.")

        self.root.columnconfigure(1, weight=2)
        self.root.columnconfigure((0, 2), weight=1)
        for i in range(7):
            self.root.rowconfigure(i, weight=1 if 0 < i < 6 else 5)

    def _create_widgets(self):
        row = 1
        # Przycisk Uruchom
        tk.Button(self.root, text="Uruchom", command=self.start_handler).grid(row=row, column=1, sticky="NSEW",
                                                                              pady=(0, 5))

        # Combobox z kamerami
        names = [c['name'] for c in self.cameras_details]
        self.camera_select = ttk.Combobox(self.root, values=names, state="readonly")
        self.camera_select.grid(row=row + 1, column=1, sticky="NSEW", pady=(0, 15))
        if names: self.camera_select.current(0)

        # Pozostałe przyciski
        tk.Button(self.root, text="Statystyka", command=self.statistic_handler).grid(row=row + 2, column=1,
                                                                                     sticky="NSEW", pady=(0, 10))
        tk.Button(self.root, text="Uczenie", command=self.learning_handler).grid(row=row + 3, column=1, sticky="NSEW",
                                                                                 pady=(0, 10))
        tk.Button(self.root, text="Wyjdź", command=self.exit_handler).grid(row=row + 4, column=1, sticky="NSEW",
                                                                           pady=(0, 10))

    def start_handler(self):
        camera_name = self.camera_select.get()
        camera_id = next((c['id'] for c in self.cameras_details if c['name'] == camera_name), 0)

        print(f"Wybrano: {camera_name} (ID: {camera_id})")

        skeleton_gen = Skeleton_generator(False )
        stream = VideoPoseStream(skeleton_gen.generate, recognize_intention,camera_id)
        stream.activate( )

    def statistic_handler(self):
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Frame): widget.destroy()

        dane = self.count_photos()
        if not dane: return

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(list(dane.values()), labels=list(dane.keys()), autopct="%1.1f%%", startangle=90)

        frame = tk.Frame(self.root, bg="white")
        frame.place(relx=0.5, rely=0.5, anchor="center")
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
        tk.Button(frame, text="Zamknij", command=frame.destroy).pack()

    def count_photos(self, folder="dataset/pictures"):
        if not os.path.exists(folder): return {}
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]
        return Counter(f.split("_")[0] for f in files)

    def learning_handler(self):
        messagebox.showinfo("info", "Przetwarzanie zdjęć...")
        Skeleton_generator(True).process_photos(r"dataset\pictures", r"dataset\processed_data")
        messagebox.showinfo("info", "Uczenie maszyny...")
        train_models()

    def exit_handler(self):
        self.root.destroy()

    def activate(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = SilhouetteApp()
    app.activate()