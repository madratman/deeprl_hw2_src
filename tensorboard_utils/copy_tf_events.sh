mkdir -p runs/vanilla_enduro
mkdir -p runs/vanilla_space
mkdir -p runs/double_enduro
mkdir -p runs/double_space

scp ratneshm@perceptron.ri.cmu.edu:/data/datasets/ratneshm/deeprl_hw2/Enduro-v0/vanilla/2017-03-28_05-27-38/events.out.tfevents.1490678859.73b0254e3e7e runs/vanilla_enduro
scp ratneshm@perceptron.ri.cmu.edu:/data/datasets/ratneshm/deeprl_hw2/SpaceInvaders-v0/double/2017-03-28_05-18-57/events.out.tfevents.1490678338.351c8a47804f runs/double_space/
scp ratneshm@perceptron.ri.cmu.edu:/data/datasets/ratneshm/deeprl_hw2/SpaceInvaders-v0/vanilla/2017-03-28_05-18-54/events.out.tfevents.1490678336.3ced6ec9fa2a runs/vanilla_space
scp ratneshm@perceptron.ri.cmu.edu:/data/datasets/ratneshm/deeprl_hw2/Enduro-v0/double/2017-03-28_05-27-45/events.out.tfevents.1490678867.dc0585ede37b runs/double_enduro

mkdir -p runs/q3
mkdir -p runs/q4
mkdir -p runs/q7
mkdir -p runs/linwithoutstuff

scp ratneshm@perceptron.ri.cmu.edu:/data/datasets/rbonatti/deeprl_hw2/q3/SpaceInvaders-v0/vanilla/2017-03-29_08-31-52/events* runs/q3
scp ratneshm@perceptron.ri.cmu.edu:/data/datasets/rbonatti/deeprl_hw2/q4/SpaceInvaders-v0/double/2017-03-29_08-31-58/events* runs/q4
scp ratneshm@perceptron.ri.cmu.edu:/data/datasets/rbonatti/deeprl_hw2/q7/SpaceInvaders-v0/vanilla/2017-03-29_08-32-01/events* runs/q7
scp ratneshm@perceptron.ri.cmu.edu:/data/datasets/rbonatti/deeprl_hw2/linwithoutstuff/SpaceInvaders-v0/vanilla/2017-03-29_08-31-52/events* runs/linwithoutstuff
