tensorboard --logdir=vanilla_enduro:runs/vanilla_enduro --port=6006& > /dev/null 2>&1
tensorboard --logdir=double_enduro:runs/double_enduro --port=6007& > /dev/null 2>&1
tensorboard --logdir=vanilla_space:runs/vanilla_space --port=6008& > /dev/null 2>&1
tensorboard --logdir=double_space:runs/double_space --port=6009& > /dev/null 2>&1
tensorboard --logdir=q3:runs/q3 --port=6010& > /dev/null 2>&1
tensorboard --logdir=q4:runs/q4 --port=6011& > /dev/null 2>&1
tensorboard --logdir=q7:runs/q7 --port=6012& > /dev/null 2>&1
tensorboard --logdir=linwithoutstuff:runs/linwithoutstuff --port=6013& > /dev/null 2>&1


# without http opens new window
google-chrome http://localhost:6006&
google-chrome http://localhost:6007&
google-chrome http://localhost:6008&
google-chrome http://localhost:6009&
google-chrome http://localhost:6010&
google-chrome http://localhost:6011&
google-chrome http://localhost:6012&
google-chrome http://localhost:6013&

