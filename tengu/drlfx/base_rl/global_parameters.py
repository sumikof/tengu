class GlobalParameters:
    batch_size = 32
    gamma = 0.99
    base_epsilon = 0.99
    multi_reward_size = 3
    num_episodes = 300
    max_steps = 200
    train_interval = 2
    learning_rate = 0.01
    hidden_size = 10
    memory_capacity = 10000
    per_alpha = 0.6
    save_weight = False
    wight_file_name = 'test_weight'


    def __init__(self):
        pass

