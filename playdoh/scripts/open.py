if __name__ == '__main__':
    from playdoh import *
    import sys
    if len(sys.argv) <= 1:
        cpu = get_cpu_count()
        gpu = get_gpu_count()
        port = DEFAULT_PORT
    else:
        cpu = int(sys.argv[1])
        gpu = int(sys.argv[2])
        port = int(sys.argv[3])
        if cpu == -1:
            cpu = get_cpu_count()
        if gpu == -1:
            gpu = get_gpu_count()
    open_server(maxcpu=cpu, maxgpu=gpu, port=port)
