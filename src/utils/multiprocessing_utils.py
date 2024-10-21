import psutil

def get_optimal_process_count(max_cpu_percent):
    cpu_count = psutil.cpu_count()
    return max(1, min(cpu_count, int(cpu_count * (max_cpu_percent / 100))))