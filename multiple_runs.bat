@echo off
set steps=250_000
set eval_int=25_000
:: set logdir=logdir/random_agent
set logdir=logdir/own_agent
start C:\Users\Antonin\anaconda3\envs\envCrafterGame\python train.py --steps %steps% --eval-interval %eval_int% --logdir %logdir%/1 
start C:\Users\Antonin\anaconda3\envs\envCrafterGame\python train.py --steps %steps% --eval-interval %eval_int% --logdir %logdir%/2 
start C:\Users\Antonin\anaconda3\envs\envCrafterGame\python train.py --steps %steps% --eval-interval %eval_int% --logdir %logdir%/3 
:: start C:\Users\Antonin\anaconda3\envs\envCrafterGame\python train.py --steps %steps% --eval-interval %eval_int% --logdir %logdir%/4
:: start C:\Users\Antonin\anaconda3\envs\envCrafterGame\python train.py --steps %steps% --eval-interval %eval_int% --logdir %logdir%/5
