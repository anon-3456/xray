import os

directory = "weight_decay_larger/"
MAX_GID = 0
MAX_FOLD = 4

# ANSI color codes
RESET = "\033[0m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
GRAY = "\033[90m"

def progress_bar(current, total, length=15):
    filled = int(length * (current + 1) / (total + 1))
    bar = "#" * filled + "-" * (length - filled)

    if current >= total:
        color = GREEN
    elif current >= total / 2:
        color = YELLOW
    else:
        color = GRAY

    return f"{color}[{bar}]{RESET} {current}/{total}"

def get_last_epoch_line(lines: list[str]):
    for i in range(len(lines)):
        if lines[~i].startswith("Epoch "):
            return lines[~i]
    return ""

for file in sorted(list(os.listdir(directory))):
    path = os.path.join(directory, file)
    if not os.path.isdir(path) or file.startswith("."):
        continue

    log_path = os.path.join(path, "logs/main.txt")
    with open(log_path, "r") as logfile:
        last_line = get_last_epoch_line(logfile.readlines())
        last_line = last_line.split("/")
        
        if len(last_line) == 2:
            epoch = int(last_line[0][-2:])
            total_epochs = int(last_line[1][:2])
        else:
            epoch = 0
            total_epochs = 30

    indices_directory = os.path.join(path, "objects/indices")
    status = sorted(os.listdir(indices_directory))[-1]
    g_id, fold = int(status[0]), int(status[7])

    g_bar = progress_bar(g_id, MAX_GID)
    f_bar = progress_bar(fold, MAX_FOLD)
    e_bar = progress_bar(epoch, total_epochs)

    print(f"{file:30s}  g_id {g_bar}   fold {f_bar}   epoch {e_bar}")
