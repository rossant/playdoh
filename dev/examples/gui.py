from Tkinter import *
from threading import Thread
import playdoh, os

def get_available_resources(server):
    try:
        obj = playdoh.get_available_resources(server)[0]
        msg = None
    except:
        obj = None
        msg = "Unable to connect to %s:%s" % server
    return obj, msg

def get_my_resources(server):
    try:
        obj = playdoh.get_my_resources(server)[0]
        msg = None
    except:
        obj = None
        msg = "Unable to connect to %s:%s" % server
    return obj, msg

def set_my_resources(server, cpu, gpu):
#    try:
    obj = playdoh.request_resources(server, CPU=cpu, GPU=gpu)
    msg = None
#    except:
#        obj = None
#        msg = "Unable to connect to %s:%s" % server
    return obj, msg




class PlaydohGUI:
    def __init__(self, master):
        self.master = master
        frame = Frame(master)
        self.frame = frame
        self.sliders = {}
        #self.total_resources = None
        self.server = None
        
        try:
            self.servers = playdoh.USERPREF['favoriteservers']
        except:
            self.servers = ['']
        
        self.textbox_server = Text(width=35, height=1)
        self.textbox_server.grid(row=0,columnspan=2)
        self.textbox_server.insert(END, '')
        
        self.yScroll = Scrollbar(master, orient=VERTICAL)
        self.yScroll.grid(row=1, column=1)
        self.listbox_servers = Listbox(width=30, height=1, yscrollcommand=self.yScroll.set,
                                       activestyle=None)
        self.yScroll["command"] = self.listbox_servers.yview
        self.listbox_servers.grid(row=1,column=0)
#        self.listbox_servers.insert(END, '')
        for server in self.servers:
            self.listbox_servers.insert(END, server)
#        self.poll() # start polling the list
        
        self.button_info = Button(master, text="Retrieve info from the server", 
                                    width=32, height=1, font="Arial 11 bold",
                                    command=self.get_info)
        self.button_info.grid(row=2,columnspan=2)
        
        self.textbox_info = Text(width=35, height=10)
        self.textbox_info.grid(row=3,columnspan=2)
        self.textbox_info.insert(END, "Idle resources:\nCPU:  \nGPU:  \n\nAllocated resources\nCPU:  \nGPU:  \n")

        Label(text="Number of CPUs").grid(row=4, column=0)
        self.set_slider('CPU', 0, row=4, column=1, callback=self.callback_cpu)

        Label(text="Number of GPUs").grid(row=5, column=0)
        self.set_slider('GPU', 0, row=5, column=1, callback=self.callback_gpu)
        
        self.button_launch = Button(master, text="Set units", width=35, height=2,
                                    foreground="blue", font="Arial 11 bold",
                                    command=self.set_units)
        self.button_launch.grid(row=6,columnspan=2)
        
        self.button_exit = Button(master, text="Exit", width=35, height=1,
                                    font="Arial 11 bold", command=self.exit)
        self.button_exit.grid(row=7,columnspan=2)

#    def poll(self):
#        now = self.listbox_servers.curselection()
#        if now != self.current:
#            self.list_has_changed(now)
#            self.current = now
#        self.after(250, self.poll)
        
    def disable_buttons(self):
        self.button_info.config(state = DISABLED)
        self.button_launch.config(state = DISABLED)
        self.button_exit.config(state = DISABLED)
   
    def enable_buttons(self):
        self.button_info['state'] = NORMAL
        self.button_launch['state'] = NORMAL
        self.button_exit['state'] = NORMAL
   
    def set_slider(self, name, units, row, column, callback):
        self.sliders[name] = Scale(self.master, from_=0, to=units,
                            command = callback,
                            orient=HORIZONTAL)
        self.sliders[name].grid(row=row,column=column)
   
    def update_sliders(self, cpu, gpu):
        self.set_slider('CPU', cpu, row=4, column=1, callback=self.callback_cpu)
        self.set_slider('GPU', gpu, row=5, column=1, callback=self.callback_gpu)
   
    def callback_cpu(self, value):
        self.cpu = int(value)

    def callback_gpu(self, value):
        self.gpu = int(value)
    
    def get_server(self):
        server = str(self.textbox_server.get(1.0, END).strip(" \n\r"))
        if server == '':
            index = self.listbox_servers.nearest(0)
            server = self.servers[index]
            if server == '':
                log_warn("No server selected")
                return
        fullserver = server
        self.server_address = server
        l = server.split(':')
        if len(l) == 1:
            server, port = l[0], str(DEFAULT_PORT)
        elif len(l) == 2:
            server, port = l[0], l[1]
        else:
            raise Exception("server IP must be 'IP:port'")
        server = server.strip()
        port = int(port.strip())
        self.server = (server, port)
        return fullserver
    
    def _get_info(self):
        self.disable_buttons()
        self.get_server()
        
        try:
            playdoh.GC.set([self.server])
            disconnect = playdoh.GC.connect()
            resources, msg = get_available_resources(self.server)
            if msg is not None: playdoh.log_warn(msg)
        
            self.textbox_info.delete("2.5", "2.6")
            self.textbox_info.insert("2.5", resources['CPU'])
            self.textbox_info.delete("3.5", "3.6")
            self.textbox_info.insert("3.5", resources['GPU'])
            self.update_sliders(resources['CPU'], resources['GPU'])
            
            resources, msg = get_my_resources(self.server)
            if msg is not None: playdoh.log_warn(msg)
            self.textbox_info.delete("6.5", "6.6")
            self.textbox_info.insert("6.5", resources['CPU'])
            self.textbox_info.delete("7.5", "7.6")
            self.textbox_info.insert("7.5", resources['GPU'])
            for r in sorted(resources.keys()):
                if resources[r] is not None:
                    self.cpu = int(resources[r])
                    self.sliders[r].set(self.cpu)
            if disconnect: playdoh.GC.disconnect()
        
        except:
            playdoh.log_warn("Unable to connect to the server")
            
        self.enable_buttons()
            
    def get_info(self):
        server = self.get_server()
        if (server not in self.servers) and (server != ''):
            self.servers.append(server)
            self.listbox_servers.insert(END, server)
            playdoh.USERPREF['favoriteservers'] = self.servers
            playdoh.USERPREF.save()
        playdoh.log_info("Connecting to %s" % server)
        if os.name != 'posix':
            Thread(target=self._get_info).start()
        else:
            self._get_info()
        
    def _set_units(self):
        self.disable_buttons()
        self.get_server()
        
        playdoh.GC.set([self.server])
        disconnect = playdoh.GC.connect()
        
        set_my_resources(self.server, self.cpu, self.gpu)
        self._get_info()
                
        if disconnect: playdoh.GC.disconnect()
        
        self.enable_buttons()
        
    def set_units(self):
        if os.name != 'posix':
            Thread(target=self._set_units).start()
        else:
            self._set_units()

    def exit(self):
        self.frame.quit()




if __name__ == '__main__':
    root = Tk()
    
    # Window resizable
    root.resizable(True, True)
    
    # Size of the window
    w = 400
    h = 480
    
    # Centers window on the screen
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    
    root.geometry("%dx%d%+d%+d" % (w, h, x, y))
    app = PlaydohGUI(root)
    root.mainloop()
