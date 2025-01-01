# gui_interface.py

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import torch
from model_testing import coreManger

class ModelTestingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Testing Interface")
        self.root.geometry("800x600")
        
        # Variables
        self.imagename_var = tk.StringVar(value="Chlorite2_after_biotite_Clip")
        self.tile_size = 256
        self.num_channels = 6
        self.num_classes = 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Model Testing Interface", 
                              font=('Helvetica', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Image name input
        ttk.Label(main_frame, text="Image Name:").grid(row=1, column=0, 
                                                      sticky=tk.W, pady=5)
        imagename_entry = ttk.Entry(main_frame, textvariable=self.imagename_var, 
                                  width=40)
        imagename_entry.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Tile size display
        ttk.Label(main_frame, text="Tile Size:").grid(row=2, column=0, 
                                                     sticky=tk.W, pady=5)
        ttk.Label(main_frame, text=str(self.tile_size)).grid(row=2, column=1, 
                                                            sticky=tk.W, pady=5)
        
        # Device info display
        ttk.Label(main_frame, text="Device:").grid(row=3, column=0, 
                                                  sticky=tk.W, pady=5)
        ttk.Label(main_frame, text=str(self.device)).grid(row=3, column=1, 
                                                         sticky=tk.W, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, length=300, 
                                          mode='determinate', 
                                          variable=self.progress_var)
        self.progress_bar.grid(row=4, column=0, columnspan=2, pady=10)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var)
        self.status_label.grid(row=5, column=0, columnspan=2, pady=5)
        
        # Run button
        self.run_button = ttk.Button(main_frame, text="Run Model Test", 
                                   command=self.run_model_thread)
        self.run_button.grid(row=6, column=0, columnspan=2, pady=10)
        
        # Log text area
        self.log_text = tk.Text(main_frame, height=10, width=50)
        self.log_text.grid(row=7, column=0, columnspan=2, pady=10)
        
    def update_progress(self, value, message):
        self.progress_var.set(value)
        self.status_var.set(message)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        
    def run_model_thread(self):
        self.run_button.state(['disabled'])
        thread = threading.Thread(target=self.run_model)
        thread.daemon = True
        thread.start()
        
    def run_model(self):
        try:
            imagename = self.imagename_var.get()
            
            # Update initial progress
            self.update_progress(0, "Starting model test...")
            
            # Run the model test with callback for progress updates
            success, message = testmodel(
                imagename=imagename,
                tile_size=self.tile_size,
                num_channels=self.num_channels,
                num_classes=self.num_classes,
                device=self.device,
                callback=self.update_progress
            )
            
            if not success:
                messagebox.showerror("Error", message)
                
        except Exception as e:
            self.update_progress(0, f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))
        finally:
            self.run_button.state(['!disabled'])

def main():
    root = tk.Tk()
    app = ModelTestingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()