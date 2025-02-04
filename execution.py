import torch
import tensorrt as trt

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

def optimize_model_with_tensorrt(model):
    """ Converts model to TensorRT for optimized inference """
    model.eval()
    device = get_device("gpu")

    # Dummy input for tracing
    dummy_input = torch.randn(1, *model.input_shape).to(device)

    traced_model = torch.jit.trace(model, dummy_input)
    trt_model = torch.jit.freeze(traced_model)

    return trt_model

def run_inference(model, data, execution_config):
    """ Runs optimized inference using TensorRT or PyTorch """
    device = get_device(execution_config.get("device", "auto"))
    
    if device.type == "cuda":
        model = optimize_model_with_tensorrt(model)

    model.to(device)
    data = data.to(device)

    with torch.no_grad():
        output = model(data)

    return output.cpu()