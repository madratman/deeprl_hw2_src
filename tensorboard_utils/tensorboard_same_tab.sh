tensorboard --logdir=vanilla_enduro:runs/vanilla_enduro, double_enduro:runs/double_enduro, vanilla_space:runs/vanilla_space, double_space:runs/double_space, linear_vanilla_with:runs/q3, linear_double_Q:runs/q4, duelling:runs/q7, linear_vanilla_WITHOUT:runs/linwithoutstuff & > /dev/null 2>&1
google-chrome http://localhost:6006&
