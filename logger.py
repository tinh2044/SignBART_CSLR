import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist

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
            self.meters[k].update(v.item() if isinstance(v, torch.Tensor) else v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        return self.delimiter.join(f"{name}: {meter}" for name, meter in self.meters.items())

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=""):
        start_time, end = time.time(), time.time()
        iter_time, data_time = SmoothedValue(fmt='{avg:.4f}'), SmoothedValue(fmt='{avg:.4f}')
        
        log_template = self._generate_log_template(header, len(iterable))
        MB = 1024.0 * 1024.0
        
        for i, obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            
            if i % print_freq == 0 or i == len(iterable) - 1:
                self._print_log(i, iterable, iter_time, data_time, log_template, MB)
            end = time.time()
        
        self._print_total_time(header, start_time, len(iterable))

    def _generate_log_template(self, header, total_iters):
        space_fmt = ':' + str(len(str(total_iters))) + 'd'
        log_msg = [
            header, f'[{"{0" + space_fmt + "}"}/{total_iters}]',
            'eta: {eta}', '{meters}', 'time: {time}', 'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        return self.delimiter.join(log_msg)

    def _print_log(self, i, iterable, iter_time, data_time, log_template, MB):
        eta_seconds = iter_time.global_avg * (len(iterable) - i)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        memory_usage = torch.cuda.max_memory_allocated() / MB if torch.cuda.is_available() else None
        
        print(log_template.format(
            i, len(iterable), eta=eta_string, meters=str(self),
            time=str(iter_time), data=str(data_time),
            memory=memory_usage if memory_usage is not None else ""
        ))

    def _print_total_time(self, header, start_time, total_iters):
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        
        print(f'{header} Total time: {total_time_str} ({total_time / total_iters:.4f} s / it)')

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, float) or isinstance(value, int):
                self.meters[key].update(value)
            else:
                raise ValueError(f"Value for {key} must be a float or int, got {type(value)}.")

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        return self.delimiter.join(f"{name}: {meter}" for name, meter in self.meters.items())

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=""):
        if print_freq == 0:
            print_freq = 1
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = f":{len(str(len(iterable)))}d"
        log_template = self.delimiter.join([
            header,
            f"[{{0{space_fmt}}}/{len(iterable)}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ])

        for i, obj in enumerate(iterable, start=1):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)

            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                print(
                    log_template.format(
                        i,
                        len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                    )
                )

            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")