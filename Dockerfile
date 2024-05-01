FROM nvcr.io/nvidia/tritonserver:24.02-py3-sdk
RUN pip install itk torch scipy dynamic_network_architectures

