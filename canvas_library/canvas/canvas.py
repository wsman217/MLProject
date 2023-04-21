"""
28*28 Pixel size
168*168 canvas size 6x aspect ratio
"""
import os

import customtkinter
import tkinter
import numpy as np
from PIL import Image as im


class Canvas:

    def save(self):
        name = self.fileNameInput.get()
        dir_list = [int(x.split(".")[0]) for x in os.listdir(os.getcwd() + "/gan/" + name)]
        name = os.getcwd() + "/gan/" + name + "/" + str(0 if len(dir_list) == 0 else max(dir_list) + 1) + ".png"
        data = im.fromarray((self.pixels * 255).astype(np.uint8))
        data.save(name)

    def paint(self, event):
        x, y = event.x, event.y
        self.place_pixel(x, y, "000")

    def erase(self, event):
        x, y = event.x, event.y
        self.place_pixel(x, y, "fff")

    def place_pixel(self, x, y, color, call_listeners=True):
        if x <= 0 or x > 167 or y <= 0 or y > 167:
            return
        x = (x - (x % 6))
        y = (y - (y % 6))

        self.pixels[int(y / 6)][int(x / 6)] = 1 if color == "000" else 0
        self.canvas.create_rectangle(x, y, x + 6, y + 6, fill="#" + color)

        if call_listeners:
            for method_to_call in self.listeners:
                method_to_call(self.pixels, self)

    def clear_canvas(self):
        self.canvas.delete("all")
        for x in range(0, 168):
            for y in range(0, 168):
                self.place_pixel(x, y, "fff", False)
        self.place_pixel(1, 1, "fff")

    def register_listener(self, method):
        self.listeners.append(method)
        return self

    def start(self):
        self.app.mainloop()
        return self

    def update_prediction_text(self, prediction):
        self.prediction_text.set(prediction)

    def __init__(self):
        customtkinter.set_appearance_mode("System")
        customtkinter.set_default_color_theme("blue")

        self.app = customtkinter.CTk()
        # +1070+100 moves the frame
        self.app.geometry("240x320")

        self.listeners = []
        self.pixels = np.empty([28, 28], dtype=int)

        self.canvas = tkinter.Canvas(self.app, width=167, height=167)
        self.canvas.pack(pady=10)

        self.clear_canvas()
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint, add="+")
        self.canvas.bind("<Button-2>", lambda ignore: self.clear_canvas())
        self.canvas.bind("<B3-Motion>", self.erase)
        self.canvas.bind("<Button-3>", self.erase, add="+")

        self.save = customtkinter.CTkButton(master=self.app, text="Save", command=self.save)
        self.save.place(anchor=tkinter.S)
        self.save.pack(pady=10)

        self.prediction_text = customtkinter.StringVar(value="-")
        self.prediction_box = customtkinter.CTkLabel(master=self.app, textvariable=self.prediction_text, width=25,
                                                     height=25)
        self.prediction_box.place(anchor=tkinter.SW)
        self.prediction_box.pack(padx=10)

        self.fileNameInput = customtkinter.StringVar()
        self.saveInput = customtkinter.CTkEntry(master=self.app, placeholder_text="File Name",
                                                textvariable=self.fileNameInput)
        self.saveInput.place(anchor=tkinter.S)
        self.saveInput.pack(pady=10)
