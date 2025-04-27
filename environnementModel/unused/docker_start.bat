docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --group-add video --ipc=host --shm-size 8G -v "../../:/app/" amdih/ryzen-ai-pytorch:latest
