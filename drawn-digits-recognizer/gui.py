import tkinter as tk
from tkinter import *
from tkinter import ttk, colorchooser
from PIL import Image, ImageDraw
import numpy as np
import tf_model as tfmod

class GUI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.weight = 300
        self.height = 300
        self.resizable(False, False)
        self.penwidth = 20
        self.previous_x = self.previous_y = 0
        self.x = self.y = 0
        self.points_recorded = []
        self.canvas = tk.Canvas(self, width=self.weight, height=self.height, bg = "white", cursor="cross")
        self.canvas.pack(side="top", fill="both", expand=True)
        self.image1 = Image.new("RGB", (self.weight, self.height), "white")
        self.draw = ImageDraw.Draw(self.image1)
        self.pred_label = Label(self, text="Draw a digit in the box!")
        self.pred_label.pack(side="left", fill="both", expand=True)
        #self.button_print = tk.Button(self, text = "Predict", command = self.predict)
        #self.button_print.pack(side="right", fill="both", expand=True)
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
        self.button_clear.pack(side="right", fill="both", expand=True)
        self.canvas.bind("<Motion>", self.tell_me_where_you_are)
        self.canvas.bind("<B1-Motion>", self.draw_from_where_you_are)
        
        self.drawing = False

    def predict(self):
        filename = "my_drawing.jpg"
        self.image1 = self.image1.resize((28, 28), Image.ANTIALIAS)
        
        #self.image1.save(filename)
        #img = Image.open( filename )
        #img.load()
        
        img = self.image1
        img = img.convert('L') # convert 3 channel (RGB) -> 1 channel (gray)
        
        data = np.asarray( img, dtype="int32" )
        data = 255 - data # swap colors
        
        #print("data-shape: ",data.shape)
        #print(data)
        
        tfm = tfmod.TFModel()
        (max_val, pred_val) = tfm.predict_image(data)
        
        msg = "Number predicted: " + str(max_val) + " (Confidence: " + str(round((pred_val * 100), 2)) + ")"
        self.pred_label['text'] = msg


    def clear_all(self):
        self.canvas.delete("all")
        self.image1 = Image.new("RGB", (self.weight, self.height), "white")
        self.draw = ImageDraw.Draw(self.image1)
        self.pred_label['text'] = "Draw a digit in the box!"

    def tell_me_where_you_are(self, event):
        self.previous_x = event.x
        self.previous_y = event.y
        
        if self.drawing:
            self.drawing = False
            self.predict()

    def draw_from_where_you_are(self, event):
        if self.points_recorded:
            self.points_recorded.pop()
            self.points_recorded.pop()

        self.x = event.x
        self.y = event.y
        self.canvas.create_line(self.previous_x, self.previous_y, self.x, self.y,fill="black", width=self.penwidth,capstyle=ROUND, smooth=True)
        self.draw.line([self.previous_x, self.previous_y, self.x, self.y], "black", width=self.penwidth)
        self.points_recorded.append(self.previous_x)
        self.points_recorded.append(self.previous_y)
        self.points_recorded.append(self.x)     
        self.points_recorded.append(self.x)        
        self.previous_x = self.x
        self.previous_y = self.y
        
        self.drawing = True
       

if __name__ == "__main__":
    app = GUI()
    app.title("Predicting written digits")
    app.mainloop()