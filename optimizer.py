from typing import Callable, Optional

import torch
from torch import nn

from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

from torch.optim import Optimizer


def build_gradient_clipper(config):
    clip_grad_fun = None
    if "clip_grad_val" in config.keys():
        clip_value = config["clip_grad_val"]
        clip_grad_fun = lambda params: nn.utils.clip_grad_value_(
            parameters=params, clip_value=clip_value
        )
    elif "clip_grad_norm" in config.keys():
        max_norm = config["clip_grad_norm"]
        clip_grad_fun = lambda params: nn.utils.clip_grad_norm_(
            parameters=params, max_norm=max_norm
        )

    if "clip_grad_val" in config.keys() and "clip_grad_norm" in config.keys():
        raise ValueError("You can only specify either clip_grad_val or clip_grad_norm.")

    return clip_grad_fun


def build_optimizer(config, model) :

    optimizer_name = config.get("optimizer", "adam").lower()
    weight_decay = config.get("weight_decay", 0)
    eps = config.get("eps", 1.0e-8)
    parameters = []
    base_lr = config['learning_rate'].pop('default')
    base_lr = float(base_lr)
    for n, p in model.named_children():
        lr_ = base_lr
        for m, lr in config['learning_rate'].items():
            if m in n:
                lr_ = lr
        parameters.append({'params': p.parameters(), 'lr': lr_})

    betas = config.get("betas", (0.9, 0.999))
    amsgrad = config.get("amsgrad", False)
    if optimizer_name == "adam":
        return torch.optim.Adam(
            params=parameters,
            lr=base_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
    elif optimizer_name == "adamw":
        return torch.optim.Adam(
            params=parameters,
            lr=base_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
    elif optimizer_name == "adagrad":
        return torch.optim.Adagrad(
            params=parameters,
            lr=base_lr,
            lr_decay=config.get("lr_decay", 0),
            weight_decay=weight_decay,
            eps=eps,
        )
    elif optimizer_name == "adadelta":
        return torch.optim.Adadelta(
            params=parameters,
            rho=config.get("rho", 0.9),
            eps=eps,
            lr=base_lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop(
            params=parameters,
            lr=base_lr,
            momentum=config.get("momentum", 0),
            alpha=config.get("alpha", 0.99),
            eps=eps,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            params=parameters,
            lr=base_lr,
            momentum=config.get("momentum", 0),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError("Unknown optimizer {}.".format(optimizer_name))


def build_scheduler(config, optimizer, scheduler_mode = 'max', hidden_size = 0):
    scheduler_name = config["scheduler"].lower()

    if scheduler_name == "plateau":
        return (
            lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode=scheduler_mode,
                verbose=False,
                threshold_mode="abs",
                factor=config.get("decrease_factor", 0.1),
                patience=config.get("patience", 10),
            ),
            "validation",
        )
    elif scheduler_name == "cosineannealing":
        return (
            lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                eta_min=config.get("eta_min", 0),
                T_max=config.get("t_max", 20),
            ),
            "epoch",
        )
    elif scheduler_name == 'warmup_cosineannealing':
        return None
    elif scheduler_name == "cosineannealingwarmrestarts":
        return (
            lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=config.get("t_init", 10),
                T_mult=config.get("t_mult", 2),
            ),
            "step",
        )
    elif scheduler_name == "decaying":
        return (
            lr_scheduler.StepLR(
                optimizer=optimizer, step_size=config.get("decaying_step_size", 1)
            ),
            "epoch",
        )
    elif scheduler_name == "exponential":
        return (
            lr_scheduler.ExponentialLR(
                optimizer=optimizer, gamma=config.get("decrease_factor", 0.99)
            ),
            "epoch",
        )
    elif scheduler_name == "noam":
        factor = config.get("learning_rate_factor", 1)
        warmup = config.get("learning_rate_warmup", 4000)
        return (
            NoamScheduler(
                hidden_size=hidden_size,
                factor=factor,
                warmup=warmup,
                optimizer=optimizer,
            ),
            "step",
        )
    elif scheduler_name == "warmupexponentialdecay":
        min_rate = config.get("learning_rate_min", 1.0e-5)
        decay_rate = config.get("learning_rate_decay", 0.1)
        warmup = config.get("learning_rate_warmup", 4000)
        peak_rate = config.get("learning_rate_peak", 1.0e-3)
        decay_length = config.get("learning_rate_decay_length", 10000)
        return (
            WarmupExponentialDecayScheduler(
                min_rate=min_rate,
                decay_rate=decay_rate,
                warmup=warmup,
                optimizer=optimizer,
                peak_rate=peak_rate,
                decay_length=decay_length,
            ),
            "step",
        )
    else:
        raise ValueError("Unknown learning scheduler {}.".format(scheduler_name))


class NoamScheduler:
    def __init__(
            self,
            hidden_size: int,
            optimizer: torch.optim.Optimizer,
            factor: float = 1,
            warmup: int = 4000,
    ):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.hidden_size = hidden_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self._compute_rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate

    def _compute_rate(self):
        """Implement `lrate` above"""
        step = self._step
        return self.factor * (
                self.hidden_size ** (-0.5)
                * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    # pylint: disable=no-self-use
    def state_dict(self):
        return None


class WarmupExponentialDecayScheduler:
    """
    A learning rate scheduler similar to Noam, but modified:
    Keep the warm up period but make it so that the decay rate can be tuneable.
    The decay is exponential up to a given minimum rate.
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            peak_rate: float = 1.0e-3,
            decay_length: int = 10000,
            warmup: int = 4000,
            decay_rate: float = 0.5,
            min_rate: float = 1.0e-5,
    ):
        """
        Warm-up, followed by exponential learning rate decay.
        :param peak_rate: maximum learning rate at peak after warmup
        :param optimizer:
        :param decay_length: decay length after warmup
        :param decay_rate: decay rate after warmup
        :param warmup: number of warmup steps
        :param min_rate: minimum learning rate
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.decay_length = decay_length
        self.peak_rate = peak_rate
        self._rate = 0
        self.decay_rate = decay_rate
        self.min_rate = min_rate

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self._compute_rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate

    def _compute_rate(self):
        """Implement `lrate` above"""
        step = self._step
        warmup = self.warmup

        if step < warmup:
            rate = step * self.peak_rate / warmup
        else:
            exponent = (step - warmup) / self.decay_length
            rate = self.peak_rate * (self.decay_rate ** exponent)
        return max(rate, self.min_rate)

    # pylint: disable=no-self-use
    def state_dict(self):
        return None


class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epochs, last_epoch=-1):
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # do not start from zero
        if self.last_epoch <= 0:
            return [base_lr * 1 / (self.total_epochs + 1e-8) for base_lr in self.base_lrs]
        else:
            return [base_lr * self.last_epoch / (self.total_epochs + 1e-8) for base_lr in self.base_lrs]

    def finish(self):
        return self.last_epoch >= self.total_epochs