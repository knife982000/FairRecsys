import torch

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available, total devices: ", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"Device {i} name: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available")
