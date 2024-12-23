import torch
print(torch.cuda.is_available())  # Should return True if CUDA is available
print(torch.cuda.current_device())  # Shows the ID of the active GPU
print(torch.cuda.get_device_name(0))  # Displays the GPU name