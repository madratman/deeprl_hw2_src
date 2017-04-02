#!/bin/bash
python /home/ratneshm/courses/deep_final/deeprl_hw2_src/dqn_atari.py --question='q2' 2>&1 &
python /home/ratneshm/courses/deep_final/deeprl_hw2_src/dqn_atari.py --question='q3' 2>&1 &
wait
