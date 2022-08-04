from algo.make_env import make
from algo.SQRL import SQRL
from algo.LBAC import LBAC
from algo.RCPO import RCPO
from algo.RSPO import RSPO

# drone_xz, kine_car, kine_car_gazebo
env_name = 'drone_xz'
# env = make('drone_xz', render=True)
env = make(env_name, render=True)
method = 'SQRL'
eval = False
test_on_robot = False

if method=='SQRL':  # to use SQRL
    if env_name=='drone_xz':
        model = SQRL(env=env, env_name='drone_xz', gamma_safe=0.99, eps_safe=0.2, DGD_constraints=True,
                     use_constraint_sampling=True, nu=5000, update_nu=True, num_eps=2000, num_unsafe_transitions=4000,
                     Q_risk_ablation=True, seed=46, eval=eval)
    elif env_name=='kine_car':
        model = SQRL(env=env, env_name='kine_car', gamma_safe=0.9, eps_safe=0.3, DGD_constraints=True,
                     use_constraint_sampling=True, nu=2000, update_nu=True, num_eps=1500, num_unsafe_transitions=20,
                     Q_risk_ablation=True, seed=143, eval=eval)
    elif env_name=='kine_car_gazebo':
        model = SQRL(env=env, env_name='kine_car_gazebo', gamma_safe=0.9, eps_safe=0.3, DGD_constraints=True,
                     use_constraint_sampling=True, nu=2000, update_nu=True, num_eps=1500, num_unsafe_transitions=20,
                     Q_risk_ablation=True, seed=143, eval=eval)
    else:
        print('environment {} not supported yet'.format(env_name))

elif method=='LBAC':  # to use LBAC
    if env_name == 'drone_xz':
        model = LBAC(env=env, env_name='drone_xz', lambda_LBAC=2000, num_unsafe_transitions=100,
                     num_eps=2500, constraint_reward_penalty=2000, seed=55, batch_size=512,
                     alpha=0.2, warm_start_num=500, eval=eval)
    elif env_name == 'kine_car':
        model = LBAC(env=env, env_name='kine_car', lambda_LBAC=1000, num_unsafe_transitions=20000,
                     num_eps=2000, constraint_reward_penalty=1000, seed=1, hidden_size=64, eval=eval)
    elif env_name=='kine_car_gazebo':
        model = LBAC(env=env, env_name='kine_car_gazebo', lambda_LBAC=1000, num_unsafe_transitions=20000,
                     num_eps=2000, constraint_reward_penalty=1000, seed=1, hidden_size=64, eval=eval)
    else:
        print('environment {} not supported yet'.format(env_name))

elif method=='RCPO':  # to use RCPO
    if env_name == 'drone_xz':
        model = RCPO(env=env, env_name='drone_xz', gamma_safe=0.99, eps_safe=0.2, lambda_RCPO=3000,
                     num_eps=3000, num_unsafe_transitions=20, seed=31, eval=eval)
    elif env_name == 'kine_car':
        model = RCPO(env=env, env_name='kine_car', gamma_safe=0.9, eps_safe=0.3, lambda_RCPO=1000,
                     num_eps=2000, num_unsafe_transitions=20, seed=31, eval=eval)
    elif env_name == 'kine_car_gazebo':
        model = RCPO(env=env, env_name='kine_car_gazebo', gamma_safe=0.9, eps_safe=0.3, lambda_RCPO=1000,
                     num_eps=2000, num_unsafe_transitions=20, seed=31, eval=eval)
    else:
        print('environment {} not supported yet'.format(env_name))

elif method=='RSPO':  # to use RSPO
    if env_name == 'drone_xz':
        model = RSPO(env=env, env_name='drone_xz', gamma_safe=0.99, eps_safe=0.2, DGD_constraints=True,
                     nu_schedule=True, nu_start=10000, num_eps=2000, num_unsafe_transitions=4000, seed=46,
                     batch_size=512, alpha=0.2, eval=eval)
    elif env_name == 'kine_car':
        model = RSPO(env=env, env_name='kine_car', gamma_safe=0.99, eps_safe=0.3, DGD_constraints=True,
                     nu_schedule=True, nu_start=5000, num_eps=1500, num_unsafe_transitions=2000, seed=143,
                     Q_risk_ablation=True ,eval=eval)
    elif env_name == 'kine_car_gazebo':
        model = RSPO(env=env, env_name='kine_car_gazebo', gamma_safe=0.99, eps_safe=0.3, DGD_constraints=True,
                     nu_schedule=True, nu_start=5000, num_eps=1500, num_unsafe_transitions=2000, seed=143,
                     Q_risk_ablation=True, eval=eval)
    else:
        print('environment {} not supported yet'.format(env_name))

else:
    print('algorithm {} not supported yet'.format(method))

if not eval:
    model.learn(2000)
else:
    model.load_model(i_episode=500)
    if not test_on_robot:
        obs = env.reset()
        for i in range(1000):
            action = model.predict(obs)
            obs, reward, done, info = env.step(action)
            # env.render()
            if done:
                print('done')
                obs = env.reset()
        env.close()
    else:
        pass