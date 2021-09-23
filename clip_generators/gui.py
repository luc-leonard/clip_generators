import tkinter as tk
from tkinter import filedialog, END

import PIL.Image
from PIL import ImageTk

from models.taming_transformers.clip_generator.dreamer import network_list
from models.taming_transformers.clip_generator.generator import load_vqgan_model
from models.taming_transformers.clip_generator.image_editor import ImageEditor
import clip

class Application(tk.Frame):
    def __init__(self, master=None, vqgan=None, clip_model=None):
        super().__init__(master)
        self.master = master
        self.vqgan = vqgan
        self.clip = clip_model
        self.image_editor: ImageEditor = None

        self.create_widgets()
        self.pack()

    def create_widgets(self):
        self.load_file_button = tk.Button(self)
        self.load_file_button["text"] = "Load file"
        self.load_file_button["command"] = self.load_file
        self.load_file_button.pack(side="bottom")


        self.prompt_field = tk.Entry(self)
        self.prompt_field.pack(side="bottom")


        self.update_image_button = tk.Button(self)
        self.update_image_button["text"] = "Load text"
        self.update_image_button["command"] = self.load_text
        self.update_image_button.pack(side="bottom")


        self.image_canvas = tk.Canvas(self, width=512, height=512)
        self.image_canvas.pack(side='top')
        self.image_canvas.bind('<Button-1>', self.update_image)

    def update_image(self, event):
        self.image_editor.edit_image(None, 0.005, 10)
        self.photo_image = ImageTk.PhotoImage(self.image_editor.get_image())
        self.image_canvas.create_image(0, 0, image=self.photo_image, anchor='nw')

    def load_text(self):
        self.image_editor.load_text(self.prompt_field.get())


    def load_file(self):
        filename = filedialog.askopenfilename(initialdir="/home/lleonard/Pictures/", title="Select file", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
        self.image_editor = ImageEditor(self.vqgan, self.clip, filename)
        self.image = self.image_editor.get_image()
        self.image.show()
        self.photo_image = ImageTk.PhotoImage(self.image)
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