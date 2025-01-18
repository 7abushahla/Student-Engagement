import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

# Constants
ENGINE_FILE_PATH = "big_model.trt"  # TensorRT engine file
INPUT_NAME = "serving_default_mobilenetv2_1.00_128_input:0"  # Input tensor name
OUTPUT_NAME = "StatefulPartitionedCall:0"  # Output tensor name
DTYPE = np.float32  # Input data type (FP32)

# Load TensorRT engine
def load_engine(engine_file_path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, "rb") as f, trt.Runtime(logger) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# Allocate buffers
def allocate_buffers(engine, context):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for tensor_name in engine:
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
        shape = context.get_tensor_shape(tensor_name)
        size = trt.volume(shape)
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append((host_mem, device_mem, tensor_name))
        else:
            outputs.append((host_mem, device_mem, tensor_name))
    return inputs, outputs, bindings, stream

# Run inference
def do_inference(context, bindings, inputs, outputs, stream):
    # Set the input data and addresses
    for host_input, device_input, tensor_name in inputs:
        cuda.memcpy_htod_async(device_input, host_input, stream)
        context.set_tensor_address(tensor_name, int(device_input))
    
    # Set the output tensor addresses
    for _, device_output, tensor_name in outputs:
        context.set_tensor_address(tensor_name, int(device_output))
    
    # Execute inference
    context.execute_async_v3(stream_handle=stream.handle)

    # Get the output data
    for host_output, device_output, _ in outputs:
        cuda.memcpy_dtoh_async(host_output, device_output, stream)

    # Synchronize the stream
    stream.synchronize()

    # Return the outputs
    return [host_output for host_output, _, _ in outputs]

# Main function
def main():
    # Load the TensorRT engine
    engine = load_engine(ENGINE_FILE_PATH)
    context = engine.create_execution_context()

    # Allocate buffers
    inputs, outputs, bindings, stream = allocate_buffers(engine, context)

    # Get input tensor shape
    input_shape = context.get_tensor_shape(INPUT_NAME)

    # Create dummy input data
    input_data = np.random.random_sample(input_shape).astype(DTYPE).ravel()
    np.copyto(inputs[0][0], input_data)

    # Run inference
    output = do_inference(context, bindings, inputs, outputs, stream)

    # Process and display output
    print("Inference output:")
    for i, out in enumerate(output):
        print(f"Output {i}: {out}")

if __name__ == "__main__":
    main()
