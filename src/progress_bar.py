import tkinter as tk
from tkinter import ttk
import time


def update_progress(progress_bar, label, value):
    # Update the progress bar and the label with the current percentage
    progress_bar['value'] = value
    label.config(text=f"{value}%")
    root.update_idletasks()  # Update the UI immediately

root = tk.Tk()
root.title("Progress Bar Example")

# Create a progress bar with a larger size
progress_bar = ttk.Progressbar(root, orient="horizontal", length=500, mode="determinate", 
                               style="TProgressbar")
progress_bar.grid(row=0, column=0, padx=10, pady=10)

# Create a label for the percentage
label = tk.Label(root, text="0%", font=("Arial", 14))
label.grid(row=1, column=0)

# Apply a custom style to increase the height of the progress bar
style = ttk.Style()
style.configure("TProgressbar", thickness=40)  # Increase the thickness of the progress bar

# Simulate a task with a for loop, where 'some_variable' determines the progress
max_value = 100
for i in range(max_value + 1):
    # Update progress bar to reflect the value of 'i'
    update_progress(progress_bar, label, i)

    time.sleep(0.1)
    
# root.mainloop()

