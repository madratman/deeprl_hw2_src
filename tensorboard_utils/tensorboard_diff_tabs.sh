tensorboard --logdir=vanilla_enduro:runs/vanilla_enduro --port=6006& > /dev/null 2>&1
tensorboard --logdir=double_enduro:runs/double_enduro --port=6007& > /dev/null 2>&1
tensorboard --logdir=vanilla_space:runs/vanilla_space --port=6008& > /dev/null 2>&1
tensorboard --logdir=double_space:runs/double_space --port=6009& > /dev/null 2>&1
# without http opens new window
google-chrome http://localhost:6006&
google-chrome http://localhost:6007&
google-chrome http://localhost:6008&
google-chrome http://localhost:6009&
