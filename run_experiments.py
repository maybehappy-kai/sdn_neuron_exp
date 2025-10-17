import subprocess
import itertools
import argparse
import os
import multiprocessing
from typing import List, Dict, Any

# ====================================================================================
# --- 1. 定义实验参数网格 (SEARCH SPACE) ---
# ====================================================================================

# 定义可用的GPU ID
AVAILABLE_GPUS = [0, 1, 2, 3]

# 通用设置
DATASETS = ['simple', 'lif', 'small', 'small_3x', 'small_30x']

# 模型专属的超参数网格
SEARCH_SPACE: List[Dict[str, Any]] = [
    {
        'model': 'tcn',
        'params': {
            'lr': [1e-3, 5e-4],
            'hidden_channels': [32, 48, 64],
            'num_levels': [4, 6, 8, 10],
            'dropout': [0.2],
            'input_kernel_size': [7, 15, 31, 53],
            'weight_decay': [1e-4, 1e-3, 1e-2],
            'fusion_mode': ['add', 'ablate']
        }
    },
    {
        'model': 's4d',
        'params': {
            'lr': [2e-3, 1e-3],
            'd_model': [64, 128, 256],
            'n_layers': [2, 4, 6],
            'd_state': [64],
            'dropout': [0.1],
            'weight_decay': [1e-4, 1e-3, 1e-2],
            'fusion_mode': ['add', 'ablate']
        }
    }
]


def generate_combinations(params_grid: Dict[str, List]) -> List[Dict[str, Any]]:
    """
    从参数字典生成所有可能的组合。
    """
    keys = params_grid.keys()
    values = params_grid.values()
    combinations = []
    for combo_values in itertools.product(*values):
        combo_dict = dict(zip(keys, combo_values))
        combinations.append(combo_dict)
    return combinations


def run_single_experiment(task_info: Dict[str, Any], gpu_queue: multiprocessing.Queue, dry_run: bool):
    """
    在单个可用GPU上运行一个实验。
    它会从队列中获取一个GPU，执行任务，然后将GPU归还给队列。
    """
    gpu_id = -1
    try:
        # 1. 从队列中申请一个可用的GPU，此操作为阻塞操作
        gpu_id = gpu_queue.get()
        command = task_info['command']
        run_str = task_info['run_str']

        # 2. 设置环境变量，让子进程只能看到我们分配给它的那块GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        print(f"[GPU {gpu_id}] STARTING: {run_str}")
        print(f"[GPU {gpu_id}] > {' '.join(command)}")

        if not dry_run:
            try:
                # 3. 执行命令 (去掉了 capture_output 和 text)
                subprocess.run(command, check=True, env=env)  # <--- 看这里！
                print(f"[GPU {gpu_id}] SUCCESS: {run_str}")
            except subprocess.CalledProcessError as e:
                # 因为不再捕获输出，所以 e.stderr 将会是 None。
                # 但错误信息已经被 subprocess 自动打印到终端了。
                print(f"!!! [GPU {gpu_id}] FAILED with exit code {e.returncode}: {run_str} !!!")
        else:
            print(f"[GPU {gpu_id}] --- DRY RUN: Command not executed. ---")

    except Exception as e:
        print(f"An unexpected error occurred in the worker process: {e}")
    finally:
        # 4. 无论成功、失败还是异常，都必须将GPU ID归还给队列
        if gpu_id != -1:
            gpu_queue.put(gpu_id)
            print(f"[GPU {gpu_id}] Released, now available.")


def main(args):
    """
    主函数，用于生成所有实验任务，并使用多进程并行执行。
    """
    # --- 1. 生成所有实验任务 ---
    all_tasks = []
    run_counter = 0

    for model_config in SEARCH_SPACE:
        model_name = model_config['model']
        param_combinations = generate_combinations(model_config['params'])
        for dataset_name in DATASETS:
            for params in param_combinations:
                run_counter += 1
                command = [
                    'python', 'train.py',
                    '--model', model_name,
                    '--dataset_name', dataset_name,
                ]
                for key, value in params.items():
                    command.append(f'--{key}')
                    command.append(str(value))

                all_tasks.append({
                    'command': command,
                    'run_str': f"RUN {run_counter}/{run_counter}: {' '.join(command)}"  # Total is updated later
                })

    # 更新总任务数
    total_runs = len(all_tasks)
    for task in all_tasks:
        task['run_str'] = task['run_str'].split('/')[0] + f"/{total_runs}:" + task['run_str'].split(':')[1]

    print("=" * 60)
    print(f"Starting experiment suite. Total runs planned: {total_runs}")
    print(f"Available GPUs for parallel execution: {AVAILABLE_GPUS}")
    print(f"Number of parallel workers: {len(AVAILABLE_GPUS)}")
    print("=" * 60)

    if not all_tasks:
        print("No experiments to run. Exiting.")
        return

    # --- 2. 设置多进程环境 ---
    # Manager用于创建可在多进程之间共享的对象，这里我们用它创建一个共享队列
    manager = multiprocessing.Manager()
    gpu_queue = manager.Queue()

    # 将所有可用的GPU ID放入队列
    for gpu_id in AVAILABLE_GPUS:
        gpu_queue.put(gpu_id)

    # 创建一个进程池，进程数量等于可用GPU的数量
    pool = multiprocessing.Pool(processes=len(AVAILABLE_GPUS))

    # --- 3. 分发任务 ---
    # 将任务列表和共享的GPU队列传递给每个工作进程
    # `zip`将每个任务和dry_run标志配对，作为参数传递给`run_single_experiment`
    task_args = [(task, gpu_queue, args.dry_run) for task in all_tasks]
    pool.starmap(run_single_experiment, task_args)

    # --- 4. 清理 ---
    pool.close()
    pool.join()

    print("\n" + "=" * 60)
    print("Experiment suite finished.")
    print("=" * 60)


if __name__ == '__main__':
    # 设置多进程启动方法为 'spawn'，这在CUDA环境中更稳定
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Parallel experiment runner for TCN/S4D neuron models.")
    parser.add_argument('--dry-run', action='store_true',
                        help="Print commands and GPU assignments without executing them.")

    args = parser.parse_args()
    main(args)