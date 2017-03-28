#! /bin/bash

#python /home/ratneshm/courses/deeprl_hw2_src/dqn_atari.py --mode='double' --env='breakout' 2>&1 &
#python /home/ratneshm/courses/deeprl_hw2_src/dqn_atari.py --mode='vanilla' --env='breakout' 2>&1 &
python /home/ratneshm/courses/deeprl_hw2_src/dqn_atari.py --mode='double' --env='enduro' 2>&1 &
python /home/ratneshm/courses/deeprl_hw2_src/dqn_atari.py --mode='vanilla' --env='enduro' 2>&1 &

wait
