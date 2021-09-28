import threading
import time
import tkinter as tk
from tkinter import filedialog, END

import PIL.Image
from PIL import ImageTk

from models.taming_transformers.clip_generator.dreamer import network_list
from models.taming_transformers.clip_generator.generator import load_vqgan_model
from models.taming_transformers.clip_generator.image_editor import ImageEditor
import clip

network = network_list()

class Application(tk.Frame):
    def __init__(self, master=None, vqgan=None, clip_model=None):
        super().__init__(master)
        self.master = master
        self.vqgan = vqgan
        self.clip = clip_model
        self.image_editor: ImageEditor = None

        self.create_widgets()
        self.pack()
        self.button_pressed = False
        threading.Thread(target=self.do_edit).start()
        self.empty_image()

    def empty_image(self):
        self.image_editor = ImageEditor(self.vqgan, self.clip, None)
        self.photo_image = ImageTk.PhotoImage(self.image_editor.get_image())
        self.image_canvas.create_image(0, 0, image=self.photo_image, anchor='nw')

    def load_model(self, event):
        index = int(self.vqgan_model_type_chooser.curselection()[0])
        model_type = self.vqgan_model_type_chooser.get(index)
        print(model_type)
        self.vqqan = load_vqgan_model(network[model_type]['config'], network[model_type]['checkpoint'])
        self.empty_image()

    def create_widgets(self):
        self.load_file_button = tk.Button(self)
        self.load_file_button["text"] = "Load file"
        self.load_file_button["command"] = self.load_file
        self.load_file_button.pack(side="bottom")

        self.prompt_field = tk.Entry(self)
        self.prompt_field.pack(side="bottom")

        self.vqgan_model_type_chooser = tk. Listbox(self, listvariable=tk.StringVar(value=('wikiart', 'imagenet')))
        self.vqgan_model_type_chooser.bind('<<ListboxSelect>>', self.load_model)
        self.vqgan_model_type_chooser.pack(side='right')

        self.update_image_button = tk.Button(self)
        self.update_image_button["text"] = "Load text"
        self.update_image_button["command"] = self.load_text
        self.update_image_button.pack(side="bottom")

        self.learning_rate = tk.DoubleVar(value=0.002)
        self.learning_rate_spinbox = tk.Entry(self, textvariable=self.learning_rate)
        self.learning_rate_spinbox.pack(side='bottom')

        self.box = tk.BooleanVar()
        self.box_cb = tk.Checkbutton(self, variable=self.box)
        self.box_cb.pack(side='right')


        self.steps = tk.IntVar(value=10)
        self.steps_spin = tk.Spinbox(self, from_=2, to=50, textvariable=self.steps)
        self.steps_spin.pack(side='bottom')

        self.image_canvas = tk.Canvas(self, width=700, height=700)
        self.image_canvas.pack(side='top')
        self.image_canvas.bind('<ButtonPress-1>', self.update_image)
        self.image_canvas.bind('<ButtonRelease-1>', self.stop_update_image)


    def do_edit(self):
        while True:
            if self.button_pressed:
                for _ in self.image_editor.edit_image(self.mouse_position if self.box.get() else None, self.steps.get()):
                    self.photo_image = ImageTk.PhotoImage(self.image_editor.get_image())
                    self.image_canvas.create_image(0, 0, image=self.photo_image, anchor='nw')
            time.sleep(0)

    def stop_update_image(self, event):
        self.button_pressed = False

    def update_image(self, event):
        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        self.mouse_position = Point(event.x, event.y)
        self.image_editor.start(float(self.learning_rate.get()))
        self.button_pressed = True

    def load_text(self):
        self.image_editor.load_text(self.prompt_field.get())


    def load_file(self):
        filename = filedialog.askopenfilename(initialdir="/home/lleonard/Pictures/", title="Select file", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))

        image = PIL.Image.open(filename)
        self.image_editor = ImageEditor(self.vqgan, self.clip, image)
        self.photo_image = ImageTk.PhotoImage(self.image_editor.get_image())
        self.image_canvas.create_image(0, 0, image=self.photo_image, anchor='nw')



def main():
    network = network_list()['imagenet']
    vqqan = load_vqgan_model(network['config'], network['checkpoint'])
    clip_model = clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to('cuda:0')
    root = tk.Tk()
    app = Application(master=root, vqgan=vqqan, clip_model=clip_model)
    app.mainloop()


if __name__ == '__main__':
    main()