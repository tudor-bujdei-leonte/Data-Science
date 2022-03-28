from keras.models import load_model
import tkinter as tk
import win32gui
from PIL import ImageGrab, ImageOps
import numpy as np
import ctypes

def main():
    # 2560 x 1600 resolution bugged in Windows packages
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
    
    model = load_model("mnist-cnn.h5")
    app = App(model)
    tk.mainloop()

def predict_digit(img, model):
    # scale, invert, and grasycale
    img = img.resize((28, 28)).convert('L')
    img = ImageOps.invert(img)
    # model input reshape and normalise
    img = np.array(img).reshape(1, 28, 28, 1) / 255.0
    
    # predict
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

class App(tk.Tk):
    def __init__(self, model):
        tk.Tk.__init__(self)
        
        self._model = model
        self.x = self.y = 0
        self.radius = 10
        
        self.canv = tk.Canvas(self, width=300, height=300, 
                              bg="white", cursor="cross")
        self.label = tk.Label(self, text="Guessing...", font=("Arial", 30))
        self.classify_btn = tk.Button(self, text="Recognise", 
                                      command=self.classify_writing)
        self.clear_btn = tk.Button(self, text="Clear", 
                                   command=self.clear_writing)
        
        self.canv.grid(row=0, column=0, pady=2, sticky="W")
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.clear_btn.grid(row=1, column=0, pady=2)
        
        self.canv.bind("<B1-Motion>", self.draw)
        
    def classify_writing(self):
        canv_handle = self.canv.winfo_id()
        img = ImageGrab.grab(win32gui.GetWindowRect(canv_handle))
        digit, acc = predict_digit(img, self._model)
        
        self.label.configure(text=f"{digit}, certainty {round(acc*100,2)}%")
    
    def clear_writing(self):
        self.canv.delete("all")
    
    def draw(self, e):
        self.x = e.x
        self.y = e.y
        
        self.canv.create_oval(self.x-self.radius, self.y-self.radius, 
                              self.x+self.radius, self.y+self.radius, 
                              fill="black")
        
if (__name__ == "__main__"):
    main()