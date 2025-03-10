import torch

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available")
