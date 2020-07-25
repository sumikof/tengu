import numpy as np
import gym
import torch

ENV = 'CartPole-v0'
GAMMA = 0.99
MAX_STEP = 200
NUM_EPISODES = 1000
NUM_PROCESSES = 16
NUM_ADVANCED_STEP = 5

value_loss_coef = 0.5
enrtropy_coef = 0.01
max_gram_norm = 0.5


class RolloutStorage:
    def __init__(self, num_steps, num_processes, obs_shape):
        self.state = np.zeros(shape=(num_steps + 1, num_processes, 4))
        self.masks = np.ones(shape=(num_steps + 1, num_processes, 1))
        self.rewards = np.zeros(shape=(num_steps, num_processes, 1))
        self.actions = np.zeros(shape=(num_steps, num_processes, 1))

        # 割引報酬和
        self.returns = np.zeros(shape=(num_steps + 1, num_processes, 1))
        self.index = 0

    def insert(self, current_obs, action, reward, mask):
        self.state[self.index + 1] = current_obs.copy()
        self.masks[self.index + 1] = mask.copy()
        self.rewards[self.index] = reward.copy()
        self.actions[self.index] = action.copy()

        self.index = (self.index + 1) % NUM_ADVANCED_STEP

    def after_update(self):
        self.state[0] = self.state[-1].copy()
        self.masks[0] = self.masks[-1].copy()

    def compute_return(self, next_value):
        '''割引報酬和を計算する'''
        self.returns[-1] = next_value
        for ad_step in reversed(range(self.rewards.shape[0])):
            self.returns[ad_step] = self.returns[ad_step + 1] * GAMMA * self.masks[ad_step + 1] + self.rewards[ad_step]


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.actor = nn.Linear(n_mid, n_out)
        self.critic = nn.Linear(n_mid, 1)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        critic_output = self.critic(h2)
        actor_output = self.actor(h2)

        return critic_output, actor_output

    def get_action(self, x):
        '''状態xから行動を確率的に求める'''
        value, actor_output = self(x)
        # dim=1 で行動の種類方向にsoftmaxを計算
        action_probs = F.softmax(actor_output, dim=1)
        action = action_probs.multinomial(num_samples=1)
        # dim=1で行動の種類方向に確率計算
        return action

    def get_value(self, x):
        '''状態xから状態価値を求める'''
        value, actor_output = self(x)
        return value

    def evaluate_actions(self, x, actions):
        '''
        状態xから
        ・状態価値
        ・実際の行動actionsのlog確率
        ・エントロピー
        を求める
        '''
        value, actor_output = self(x)

        log_probs = F.log_softmax(actor_output, dim=1)

        action_log_probs = log_probs.gather(1, actions)

        probs = F.softmax(actor_output, dim=1)
        entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, entropy


import torch
from torch import optim


class Brain:
    def __init__(self, actor_critic):
        self.actor_critic = actor_critic

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.01)

    def get_actioin(self, state):
        state_tensor = torch.from_numpy(state).float()
        action_tensor = self.actor_critic.get_action(state_tensor)
        return action_tensor.detach().numpy().copy()

    def get_value(self, state):
        state_tensor = torch.from_numpy(state).float()
        value_tensor = self.actor_critic.get_value(state_tensor)
        return value_tensor.detach().numpy().copy()

    def update(self, rollouts):
        '''Advantageで計算した５つのstepを全て使って更新'''
        ##obs_shape = rollouts.state.size()[2:]  # torch.Size([4,84,84])
        num_steps = NUM_ADVANCED_STEP
        num_processes = NUM_PROCESSES

        ##x = rollouts.state[:-1].view(-1, 4)
        # x = rollouts.state[:-1].reshape(shape=(-1, 4))
        x = torch.from_numpy(rollouts.state[:-1]).float()
        x = x.view(-1, 4)

        ##actions = rollouts.actions.view(-1, 1)
        # actions = rollouts.actions.reshape(shape=(-1, 1))
        actions = torch.from_numpy(rollouts.actions).long()
        actions = actions.view(-1, 1)

        values, action_log_probs, entropy = self.actor_critic.evaluate_actions(x, actions)
        # rollouts.observations[:-1].view(-1, 4)  -> torch.Size([80, 4])
        # rollouts.actions.view(-1, 1) -> torch.Size([80, 1])
        # values -> torch.Size([80,1])
        # action_log_probs -> torch.Size([80,1])
        # entropy -> torch.Size([])
        values = values.view(num_steps, num_processes, 1)  # torch.Size([5, 16, 1])
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        # advantage (行動価値-状態価値)の計算
        returns = torch.from_numpy(rollouts.returns[:-1])
        advantages = returns - values  # torch.Size([5,16,1])

        # criticのlossを計算
        value_loss = advantages.pow(2).mean()

        # Actorのgainを計算、あとでマイナスを掛けてlossにする
        action_gain = (action_log_probs * advantages.detach()).mean()
        # detachしてadvantagesを定数にする

        # 誤差関数の総和
        total_loss = (value_loss * value_loss_coef - action_gain - entropy * enrtropy_coef)
        # 結合パラメータを更新
        self.actor_critic.train()  # 訓練モードに変更
        self.optimizer.zero_grad()  # 勾配をリセット
        total_loss.backward()  # バックプロパゲーションを計算
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_gram_norm)
        # 一気に結合パラメータが変化しすぎないように勾配の大きさは最大0.5までにする？
        self.optimizer.step()  # 結合パラメータを更新


import copy


class Environment:
    def run(self):
        '''メイン'''

        # 同時実行する環境分envを生成
        envs = [gym.make(ENV) for i in range(NUM_PROCESSES)]

        # 全エージェントが共有するBrainを生成
        n_in = envs[0].observation_space.shape[0]  # 状態4
        n_out = envs[0].action_space.n  # action 2
        n_mid = 32
        actor_critic = Net(n_in, n_mid, n_out)
        global_brain = Brain(actor_critic)

        obs_shape = n_in
        current_obs = np.zeros(shape=(NUM_PROCESSES, obs_shape))
        rollouts = RolloutStorage(
            NUM_ADVANCED_STEP, NUM_PROCESSES, obs_shape
        )
        episode_rewards = np.zeros(shape=[NUM_PROCESSES, 1])
        final_rewards = np.zeros(shape=[NUM_PROCESSES, 1])
        state_np = np.zeros([NUM_PROCESSES, obs_shape])
        reward_np = np.zeros([NUM_PROCESSES, 1])
        done_np = np.zeros([NUM_PROCESSES, 1])
        each_step = np.zeros(NUM_PROCESSES)

        episode = 0

        obs = [envs[i].reset() for i in range(NUM_PROCESSES)]

        obs = np.array(obs)
        ## obs = torch.from_numpy(obs).float()  # torch.Size([16, 4])
        current_obs = obs

        rollouts.state[0] = current_obs.copy()

        for j in range(NUM_EPISODES * NUM_PROCESSES):

            for step in range(NUM_ADVANCED_STEP):
                # 行動を求める
                with torch.no_grad():
                    action = global_brain.get_actioin(rollouts.state[step])

                # (16,1) -> (16,) -> tensor をnumpyに
                # actions = action.squeeze(1).numpy()
                actions = action.reshape((NUM_PROCESSES,))

                for i in range(NUM_PROCESSES):
                    state_np[i], reward_np[i], done_np[i], _ = envs[i].step(actions[i])

                    if done_np[i]:
                        # 環境０の時のみ出力
                        if i == 0:
                            print('{} Ebisode: Finished after {} steps'.format(episode, each_step[i] + 1))
                            episode += 1

                        # 報酬の設定
                        if each_step[i] < 195:
                            reward_np[i] = -1.0
                        else:
                            reward_np[i] = 1.0  # 立ったままなら報酬は1

                        each_step[i] = 0  # stepのリセット
                        state_np[i] = envs[i].reset()

                    else:
                        reward_np[i] = 0.0
                        each_step[i] += 1
                ## reward = torch.from_numpy(reward_np).float()
                reward = reward_np
                episode_rewards += reward

                ##masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done_np])
                masks = np.array([[0.0] if done_ else [1.0] for done_ in done_np])

                # 最後の試行のそう報酬を更新
                final_rewards *= masks  # 継続中の場合は１を掛け算してそのまま、doneの時には０を掛け算してリセット

                # 継続中は０をたす、doneはepisode_rewardsをたす
                final_rewards += (1 - masks) * episode_rewards

                # 試行の総報酬を更新
                episode_rewards *= masks  # 継続中のmaskは１、doneの場合は０

                # 現在の状態をdoneのときは全部0
                current_obs *= masks

                # current_obsお更新
                ##obs = torch.from_numpy(state_np).float()  # torch.Size([16,4])
                current_obs = state_np

                # メモリオブジェクトに今stepのtransitionを挿入
                ##rollouts.insert(current_obs, action.data, reward, masks)
                rollouts.insert(current_obs, action, reward, masks)

            with torch.no_grad():
                ##next_value = global_brain.get_value(rollouts.state[-1]).detach()
                next_value = global_brain.get_value(rollouts.state[-1])

            rollouts.compute_return(next_value)

            global_brain.update(rollouts)

            rollouts.after_update()

            ##if final_rewards.sum().numpy() >= NUM_PROCESSES:
            if final_rewards.sum() >= NUM_PROCESSES:
                print("連続成功")
                break


def test():
    cart_pole_env = Environment()
    cart_pole_env.run()


if __name__ == '__main__':
    test()
    mask = np.array([[0.0] if i % 2 == 0 else [1.0] for i in range(5)])
    print(mask)
    print(1 - mask)
