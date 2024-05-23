import  tkinter as  tk
from tkinter.filedialog import  *
from tkinter import  ttk;
#ttk模块提供对Tk 8.5中引入的Tk主题窗口小部件集的访问,几个ttk小部件（Button，Checkbutton，Entry，Frame，Label，LabelFrame，Menubutton，PanedWindow，Radiobutton，Scale和Scrollbar）将自动替换Tk小部件
import predict
import  cv2
from PIL import  Image,ImageTk
import threading
import time

class carWindow(ttk.Frame):
    pic_path = ""
    viewHigh = 600
    viewWide = 600
    updataTime = 0
    thread = None
    threadRun = False
    camera = None
    colorTransform = {"green":("绿","#55ff55"),"yello":("黄","#ffff00"),"blue":("蓝","#6666ff")}

    def __init__(self,win):
        ttk.Frame.__init__(self,win)
        frame_left = ttk.Frame(self)
        frame_right1 = ttk.Frame(self)
        frame_right2= ttk.Frame(self)
        win.title("车牌识别系统")
        win.state("normal")
        self.pack(fill=tk.BOTH,expand=tk.YES,padx="5",pady="5")
        frame_left.pack(side=LEFT,expand=1,fill=BOTH)
        frame_right1.pack(side=TOP, expand=1, fill=tk.Y)
        frame_right2.pack(side=RIGHT, expand=0)
        ttk.Label(frame_left, text='原图：').pack(anchor="nw")
        ttk.Label(frame_right1, text='车牌区域：').grid(column=0, row=0, sticky=tk.W)
        # 点击打开图片按钮，执行读取图片，并显示图片
        from_pic_ctl = ttk.Button(frame_right2, text="打开图片", width=20, command=self.from_pic)

        self.image_ctl = ttk.Label(frame_left)
        self.image_ctl.pack(anchor="nw")

        self.roi_ctl = ttk.Label(frame_right1)
        self.roi_ctl.grid(column=0, row=1, sticky=tk.W)
        ttk.Label(frame_right1, text='识别结果：').grid(column=0, row=2, sticky=tk.W)
        self.r_ctl = ttk.Label(frame_right1, text="")
        self.r_ctl.grid(column=0, row=3, sticky=tk.W)
        self.color_ctl = ttk.Label(frame_right1, text="", width="20")
        self.color_ctl.grid(column=0, row=4, sticky=tk.W)
        from_pic_ctl.pack(anchor="se", pady="5")
        self.predictor = predict.CardPredictor()   #创建识别模型
        self.predictor.train_svm()                 #训练模型

    def from_pic(self):
        self.threadRun = False
        self.pic_path = askopenfilename(title="选择识别图片",filetypes=[("jpg图片","*.jpg")])
        if self.pic_path:
            img_bgr = predict.imreadex(self.pic_path)
            self.imgtk = self.get_imgtk(img_bgr)
            self.image_ctl.configure(image=self.imgtk)
            r,roi,color = self.predictor.predict(img_bgr)  #用训练的模型进行识别
            self.show_roi(r,roi,color)

    def show_roi(self,r,roi,color):
        if r :
            roi = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
            roi = Image.fromarray(roi)
            self.imgtk_roi = ImageTk.PhotoImage(image=roi)
            self.roi_ctl.configure(image=self.imgtk_roi, state='enable')
            self.r_ctl.configure(text=str(r))
            self.updataTime = time.time()
            try:
                c = self.colorTransform[color]
                self.color_ctl.configure(text=c[0], background=c[1], state='enable')
            except Exception as e:
                print(e)
                self.color_ctl.configure(state='disabled')
        elif   self.updataTime + 8 < time.time():
                self.roi_ctl.configure(state='disabled')
                self.r_ctl.configure(text="")
                self.color_ctl.configure(state='disabled')


    def get_imgtk(self,img_bgr):
        img = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)#OpenCV是BGR格式，PIL是RGB
        im = Image.fromarray(img)  #PIL中的Image和numpy中的数组array相互转换
        imgtk = ImageTk.PhotoImage(image=im)
        wide = imgtk.width()
        high = imgtk.height()
        if wide>self.viewWide or high > self.viewHigh:
            wide_factor = self.viewWide / wide
            high_factor = self.viewHigh / high
            factor = min(wide_factor,high_factor)
            wide = int(wide*factor)
            if wide <=0 : wide = 1
            high = int(high*factor)
            if high <= 0:high = 1
            im = im.resize((wide,high),Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=im)
        return imgtk

def close_carWindow():
    print("destroy")
    if carWindow.threadRun :
        carWindow.threadRun = False
        carWindow.thread.join(2.0)
    win.destroy()


if __name__ == '__main__':
    win = tk.Tk()
    carWindow = carWindow(win)
    win.protocol('WM_DELETE_WINDOW', close_carWindow)
    win.mainloop()