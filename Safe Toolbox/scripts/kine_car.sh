#!/bin/bash

# Recovery RL (model-free recovery)
# Recovery policy: DDPG
# Task policy: 
# for i in {1..1}
# do
# 	echo "RRL MF Run $i"
# 	python -m rrl_main --cuda --env-name kine_car --use_recovery\
# 	 --MF_recovery --gamma_safe 0.8 --eps_safe 0.3 --logdir kine_car\
# 	  --logdir_suffix RRL_MF --num_eps 1000 --num_unsafe_transitions 20000 --seed $i
# done

# # Recovery RL (model-based recovery)
# for i in {1..10}
# do
# 	echo "RRL MB Run $i"
# 	python -m rrl_main --cuda --env-name kine_car --use_recovery --gamma_safe 0.8 --eps_safe 0.3 --logdir kine_car --logdir_suffix RRL_MB --num_eps 400 --num_unsafe_transitions 20000 --seed $i
# done

# Unconstrained
# for i in {1..1}
# do
# 	echo "Unconstrained Run $i"
# 	python -m rrl_main --env-name kine_car --cuda --logdir kine_car --logdir_suffix unconstrained --num_eps 4000 --num_unsafe_transitions 20000 --seed $i
# done

# Lagrangian Relaxation
# for i in {1..1}
# do
# 	echo "LR Run $i"
# 	python -m rrl_main --cuda --env-name kine_car --gamma_safe 0.8 --eps_safe 0.3\
#        --DGD_constraints --nu 5000 --update_nu --logdir kine_car --logdir_suffix LR\
#         --num_eps 1000  --num_unsafe_transitions 20 --seed $i --method-name LR --hidden_size 64
# done

#RSPO
#for i in {143..145}
#do
#	echo "RSPO Run $i"
#	python -m rrl_main --cuda --env-name kine_car --gamma_safe 0.99 \
#    --eps_safe 0.3 --DGD_constraints --nu_schedule --nu_start 5000 --logdir kine_car \
#    --logdir_suffix RSPO --num_eps 1500 --num_unsafe_transitions 2000 --seed $i --method-name RSPO --Q_risk_ablation True
#done

# RCPO
# for i in {31..35}
# do
# 	echo "RCPO Run $i"
# 	python -m rrl_main --cuda --env-name kine_car --gamma_safe 0.9\
#        --eps_safe 0.3 --RCPO --lambda_RCPO 1000 --logdir kine_car --logdir_suffix RCPO\
#         --num_eps 2000 --num_unsafe_transitions 20 --seed $i --method-name RCPO
# done

#SQRL
#for i in {143..145}
#do
#	echo "SQRL Run $i"
#	python -m rrl_main --cuda --env-name kine_car --gamma_safe 0.9 --eps_safe 0.3\
#       --DGD_constraints --use_constraint_sampling --nu 2000 --update_nu --logdir kine_car\
#        --logdir_suffix SQRL --num_eps 1500 --num_unsafe_transitions 20 --seed $i --method-name SQRL --Q_risk_ablation True
#done

# Reward Penalty
# for i in {1..4}
# do
# 	echo "RP Run $i"
# 	python -m rrl_main --env-name kine_car --cuda --constraint_reward_penalty 1000\
#         --logdir kine_car --logdir_suffix RP --num_eps 2000\
#          --num_unsafe_transitions 20000 --seed $i --method-name RP --hidden_size 64
# done


 LBAC
 for i in {1..2}
 do
 	echo "LBAC Run $i"
 	python -m rrl_main --cuda --env-name kine_car \
      --LBAC --lambda_LBAC 1000 --logdir kine_car --logdir_suffix LBAC\
       --num_eps 2000 --num_unsafe_transitions 20000\
       --constraint_reward_penalty 1000 --method-name LBAC --seed $i --hidden_size 64
 done
