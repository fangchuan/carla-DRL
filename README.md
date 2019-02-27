# carla-DRL
Implementation of about autonomous vehicle using Deep Reinforcement Learning based on CARLA simulator
2019-02-27: 修订carla_env.py: 调整camera position; 删除GROUND_Z, 正确使用Planner;
            client.read_data()得到的image数据就是处理过的数据，但是save_to_disk()时有转换为RGB图像保存
