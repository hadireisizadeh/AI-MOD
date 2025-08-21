import multiprocessing
import os

def worker(work_queue):
    while True:
        payload = work_queue.get()
        if payload == 'STOP':
            work_queue.put('STOP')
            return
        func, args, kwargs, task_name, target_path_list = payload
        if target_path_list:
            if all([os.path.exists(p) for p in target_path_list]):
                print(f"WORKER_LOG,skipped,{task_name},all files exist: skipping execution")
                return
        if not args:
            args = []
        if not kwargs:
            kwargs = {}
        try:
            func(*args, **kwargs)
            print(f"WORKER_LOG,success,{task_name},success")
        except Exception as e:
            print(f"WORKER_LOG,failure,{task_name},{str(e)}")
