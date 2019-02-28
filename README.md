// Implementation of about autonomous vehicle using Deep Reinforcement Learning based on CARLA simulator


2019-02-27: 修订carla_env.py: 调整camera position; 删除GROUND_Z, 正确使用Planner;
            client.read_data()得到的image数据就是处理过的数据，但是save_to_disk()时有转换为RGB图像保存

2019-02-28: 移植dqn代码--dqn_test.py
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
                        q_function/fully_connected/Relu [2, 6, 6, 64]