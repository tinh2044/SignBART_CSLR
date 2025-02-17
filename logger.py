import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist

import time
import datetime
import os
import torch
from collections import defaultdict
from tqdm import tqdm
from utils import is_dist_avail_and_initialized

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)



class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int)), f"Value for {k} must be float or int, got {type(v)}"
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        return self.delimiter.join(f"{name}: {meter}" for name, meter in self.meters.items())

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        if not header:
            header = ""
        start_time = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")

        # Kiểm tra có dùng GPU không
        use_cuda = torch.cuda.is_available()
        MB = 1024.0 * 1024.0

        # Kiểm tra local rank để log đúng process
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        is_main_process = local_rank in [-1, 0]

        # Tạo tqdm progress bar
        pbar = tqdm(iterable, desc=header, leave=True, disable=not is_main_process)

        end = time.time()
        for i, obj in enumerate(pbar):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)

            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                log_msg = f"eta: {eta_string} {self} time: {iter_time} data: {data_time}"

                if use_cuda:
                    max_mem = torch.cuda.max_memory_allocated() / MB
                    log_msg += f" max mem: {max_mem:.0f}"

                # Cập nhật tqdm bar
                pbar.set_postfix_str(log_msg)

            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        if is_main_process:
            print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")
