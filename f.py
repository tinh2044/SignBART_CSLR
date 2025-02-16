batch_size = 8
epochs = 100
finetune = ""
device = "cpu"
seed = 0
resume = ""
start_epoch = 0
test_on_last_epoch = "False"
num_workers = 0
pin_mem = "--pin-mem"
cfg_path = "configs/phoenix-2014t.yaml"

# Construct the Python command as a string
command_str = (
    f"python -m main "
    f"--batch-size {batch_size} "
    f"--epochs {epochs} "
    f"--finetune {finetune} "
    f"--device {device} "
    f"--seed {seed} "
    f"--resume {resume} "
    f"--start_epoch {start_epoch} "
    f"--eval "
    f"--test_on_last_epoch {test_on_last_epoch} "
    f"--num_workers {num_workers} "
    f"{pin_mem} "
    f"--cfg_path {cfg_path}"
)
print(command_str)