#!/bin/bash

# Recovery RL (model-free recovery)
# for i in {1..1}
# do
# 	echo "RRL MF Run $i"
# 	python -m rrl_main --cuda --env-name drone_xz --use_recovery\
# 	 --MF_recovery --gamma_safe 0.8 --eps_safe 0.3 --logdir drone_xz\
# 	  --logdir_suffix RRL_MF --num_eps 400 --num_unsafe_transitions 20000 --seed $i
# done

# # Recovery RL (model-based recovery)
# for i in {1..10}
# do
# 	echo "RRL MB Run $i"
# 	python -m rrl_main --cuda --env-name drone_xz --use_recovery --gamma_safe 0.8 --eps_safe 0.3 --logdir drone_xz --logdir_suffix RRL_MB --num_eps 400 --num_unsafe_transitions 20000 --seed $i
# done

# # Unconstrained
# for i in {1..10}
# do
# 	echo "Unconstrained Run $i"
# 	python -m rrl_main --env-name drone_xz --cuda --logdir drone_xz --logdir_suffix unconstrained --num_eps 400 --num_unsafe_transitions 20000 --seed $i
# done

# # Lagrangian Relaxation
# for i in {1..10}
# do
# 	echo "LR Run $i"
# 	python -m rrl_main --cuda --env-name drone_xz --gamma_safe 0.8 --eps_safe 0.3 --DGD_constraints --nu 5000 --update_nu --logdir drone_xz --logdir_suffix LR --num_eps 400 --num_unsafe_transitions 20000 --seed $i
# done

#111kaishishi genggaihuanjinghouzaicixunlian



SQRL
for i in {46..47}
do
	echo "SQRL Run $i"
	python -m rrl_main --cuda --env-name drone_xz --gamma_safe 0.99 --eps_safe 0.2 \
    --DGD_constraints --use_constraint_sampling --nu 5000 --update_nu --logdir drone_xz\
    --logdir_suffix SQRL --num_eps 2000 --num_unsafe_transitions 4000 --seed $i --method-name SQRLv1\
      --Q_risk_ablation True
done

# RSPO
# for i in {46..47}
# do
# 	echo "RSPO Run $i"
# 	python -m rrl_main --cuda --env-name drone_xz --gamma_safe 0.99 --eps_safe 0.2\
#     --DGD_constraints --nu_schedule --nu_start 10000 --logdir drone_xz --logdir_suffix RSPO\
#     --num_eps 2000 --num_unsafe_transitions 4000 --seed $i\
#     --batch_size 512 --alpha 0.2 --method-name RSPO 
#      --Q_risk_ablation True
# done
# Reward Penalty
# for i in {1..1}
# do
# 	echo "RP Run $i"
# 	python -m rrl_main --env-name drone_xz --cuda --constraint_reward_penalty 1200 \
#   --logdir drone_xz --logdir_suffix RP --num_eps 4000 --num_unsafe_transitions 100\
#    --hidden_size 64 --batch_size 512 --seed $i --method-name RP
# done

# RCPO
# for i in {31..35}
# do
# 	echo "RCPO Run $i"
# 	python -m rrl_main --cuda --env-name drone_xz --gamma_safe 0.99 --eps_safe 0.2 --RCPO \
#     --lambda_RCPO 3000 --logdir drone_xz --logdir_suffix RCPO --num_eps 3000\
#     --num_unsafe_transitions 20 --seed $i --method-name RSPO
# done

# LBAC changed
# for i in {1..1}
# do
# 	echo "LBAC Run $i"
# 	python -m rrl_main --cuda --env-name drone_xz \
#      --LBAC --lambda_LBAC 1000 --logdir drone_xz --logdir_suffix LBAC --num_unsafe_transitions 100\
#       --num_eps 100000 --constraint_reward_penalty 2000 --method-name LBAC --seed $i\
#     --batch_size 512 --alpha 0.2 --warm_start_num 2000
# done

# for i in {55..55}
# do
# 	echo "LBAC Run $i"
# 	python -m rrl_main --cuda --env-name drone_xz \
#      --LBAC --lambda_LBAC 2000 --logdir drone_xz --logdir_suffix LBAC --num_unsafe_transitions 100\
#       --num_eps 2500 --constraint_reward_penalty 2000 --method-name LBACv1 --seed $i\
#     --batch_size 512 --alpha 0.2 --warm_start_num 500
# done

# for i in {411..411}
# do
# 	echo "LBAC Run $i"
# 	python -m rrl_main --cuda --env-name drone_xz \
#      --LBAC --lambda_LBAC 2000 --logdir drone_xz --logdir_suffix LBAC --num_unsafe_transitions 100\
#       --num_eps 2000 --constraint_reward_penalty 2000 --method-name LBACv1 --seed $i\
#     --batch_size 512 --alpha 0.2 --warm_start_num 50\
#     # --hidden_size 64
# done

