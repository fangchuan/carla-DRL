// Implementation of about autonomous vehicle using Deep Reinforcement Learning based on CARLA simulator


2019-02-27: 修订carla_env.py: 调整camera position; 删除GROUND_Z, 正确使用Planner;
            client.read_data()得到的image数据就是处理过的数据，但是save_to_disk()时有转换为RGB图像保存

2019-02-28: 添加DEBUG_PRINT()功能函数

            移植dqn代码--dqn_test.py
            q_func网络 implementation_1: conv1(11x11x32,s=4,p=SAME)
                                        pool1(3x3,s=2,p=VALID)
                                        conv2(5x5x64,s=2,p=SAME)
                                        pool2(3x3, s=2,p=VALID)
                                        conv3(3x3x192,s=1,p=SAME)
                                        fc1(64)
                                        fc2(9)
                        网络结构:
                        observation shape: (224, 224, 6)--> [2,224,224,3]
                        q_function/Conv/Relu [2, 56, 56, 32]
                        q_function/MaxPool2D/MaxPool [2, 27, 27, 32]
                        q_function/Conv_1/Relu [2, 14, 14, 64]
                        q_function/MaxPool2D_1/MaxPool [2, 6, 6, 64]
                        q_function/Conv_2/Relu [2, 6, 6, 192]

2019-03-02: 测试224x224x2深度图送入网络，将会吃尽16g内存
            将图片改为84x84x2深度图, 最终占用内存7.3g， 但是carlaUE4占用内存持续增大，暂时不知道为什么.此时训练到670episode, mean_100episode_reward=1



2019-03-03: 为dqn_test.py添加load_variables,save_variables
            calculate_reward计算不合理，应该重新寻找一种reward制定模式

2019-03-04: 修改reset_env(), 每次[start_position, end_position]都随机选择Straight_poses_task中的元素
            修改calculate_reward()函数
            为每个文件添加函数说明

2019-03-05: 使用logger文件，plot csv文件观察训练过程
            想使用MPI, 发现baselines中除了dqn和trpo_mpi，其他算法训练过程均使用了mpi来vectorize environnment.
            解决Carla.sh一直吃内存问题
            calculate_reward()仍然不合理，目前参数训练16W步，mean_100ep_reward在[25,29](最高奖励100)

2019-03-06: calculate_reward方法好像不适用, 学习50W步后mean_100ep_reward=8., 出现啥问题了？
            用回原始calculate_reward方法, 把epsilon exploration step设为100000, 使用grad_norm_clip和param_noise, fc1设为512units,agent学到了不理想的行为, 10W步后仍在右转;
            
2019-03-10: DQN在单个直线任务场景中表现很好, 但是随机选择直线任务的方式来训练时，一直不能收敛， 可能是number_total_steps不够.
            尝试使用parameter_noise, 发现有问题, agent老是停在原地不动作, 需要调节参数;

2019-03-11: 还是最初的reward等式效果更好一点, 至少mean_100ep_reward能反映agent的回报趋势

2019-03-13: 将image存储格式改为np.uint8, 使用depth_image, replay_buffer为50000, dqn_test测试总共占用内存3.2G
            修改action_space为Box(shape=(4,));
            ddpg critic网络中, 将action_state与observation经过神经网络(最后一层为fc)后的vector拼接在一起;
          
2019-03-14: 将observation改成(0,255) uint8类型后, dqn很容易出现过拟合现象, 跑一会儿mean_reward就下降;
            目前dqn网络参数总量220W
            ddpg网络参数总量 3434562+ 6744720
            
2019-03-15: 把DDPG用的网络换成自己定义的网络,每个conv层和fc层都添加layer_norm,参数量=488002+849828;
            DDPG的memory换成utils/replay_buffer中的ReplayBuffer,大大降低内存占用;
