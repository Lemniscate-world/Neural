import torch

def get_device(preferred_device="auto"):
    """ Selects the best available device: GPU, CPU, or future accelerators """
    if preferred_device.lower() == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    elif preferred_device.lower() == "cpu":
        return torch.device("cpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def run_inference(model, data, execution_config):
    """ Runs inference on the specified device """
    device = get_device(execution_config.get("device", "auto"))
    model.to(device)
    data = data.to(device)

    with torch.no_grad():
        output = model(data)

    return output.cpu()
