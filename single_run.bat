@echo off
set steps=250_000
set eval_int=25_000
C:\Users\Antonin\anaconda3\envs\envCrafterGame\python train.py --steps %steps% --eval-interval %eval_int% --logdir logdir/ddqn_agent/1