import numpy as np
import copy

ENV = 'CartPole-v0'
GAMMA = 0.99
MAX_STEP = 200
NUM_EPISODES = 1000

value_loss_coef = 0.5
enrtropy_coef = 0.01
max_gram_norm = 0.5


class RolloutStorage:
    def __init__(self, num_steps, num_processes):
        self.num_steps = num_steps
        self.state = [
            [None for _ in range(num_processes)]
            for _ in range(num_steps + 1)
        ]
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

        self.index = (self.index + 1) % self.num_steps

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


class A2CNet(nn.Module):
    def __init__(self, num_actions, n_mid=32):
        super(A2CNet, self).__init__()
        input_size = STATE_SIZE * STATE_SIZE
        self.fc1 = nn.Linear(input_size, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.actor = nn.Linear(n_mid, num_actions)
        self.critic = nn.Linear(n_mid, 1)

    @classmethod
    def build(cls, builder):
        return A2CNet(num_actions=builder.test.num_actions)

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
        # action = action_probs.multinomial(num_samples=1)
        # dim=1で行動の種類方向に確率計算
        return action_probs

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


class A2CBrain:
    def __init__(self, actor_critic, num_processes, num_advanced_steps):
        self.actor_critic = actor_critic
        self.num_processes = num_processes
        self.num_advanced_steps = num_advanced_steps
        self.state_size = STATE_SIZE * STATE_SIZE
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.01)

    @classmethod
    def build(cls, builder):
        return A2CBrain(
            actor_critic=builder.build_network(),
            num_processes=builder.args.get('num_processes', 16),
            num_advanced_steps=builder.args.get('num_steps', 5)
        )

    def get_actioin(self, state):
        action_masks = np.array([mask for rates, position, current_rate, mask in state])
        input_state = np.array([rates.reshape(self.state_size) for rates, position, current_rate, mask in state])
        input_state = input_state.reshape(self.num_processes, self.state_size)
        state_tensor = torch.from_numpy(input_state).float()

        with torch.no_grad():
            action_tensor = self.actor_critic.get_action(state_tensor)
            actions_np = action_tensor.detach().numpy().copy()
        actions = np.array([[np.random.choice([0, 1, 2, 3] * mask, p=i)] for i, mask in zip(actions_np, action_masks)])
        return actions

    def get_value(self, state):
        input_state = np.array([rates.reshape(self.state_size) for rates, position, current_rate, mask in state])
        input_state = input_state.reshape(self.num_processes, self.state_size)
        with torch.no_grad():
            state_tensor = torch.from_numpy(input_state).float()
            value_tensor = self.actor_critic.get_value(state_tensor)
        return value_tensor.detach().numpy().copy()

    def update(self, rollouts):
        """Advantageで計算した５つのstepを全て使って更新"""

        num_steps = self.num_advanced_steps
        num_processes = self.num_processes

        state_list = np.array(rollouts.state[:-1])

        input_state = np.array(
            [[rates.reshape(self.state_size) for rates, position, current_rate, mask in state] for state in
             state_list])
        state = torch.from_numpy(input_state).float()
        state = state.view(-1, self.state_size)

        actions = torch.from_numpy(rollouts.actions).long()
        actions = actions.view(-1, 1)

        values, action_log_probs, entropy = self.actor_critic.evaluate_actions(state, actions)

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


class A2CAgent:
    def __init__(self):
        self.episode_rewards = np.zeros(1)
        self.final_rewards = np.zeros(1)
        self.state = None
        self.reward = np.zeros(1)
        self.done = np.zeros(1)
        self.step = np.zeros(1)

    @classmethod
    def build(cls, builder):
        return A2CAgent()


class A2CEnvironment:
    def __init__(self, agents, envs, global_brain, num_processes, num_steps, num_episodes, max_steps):
        self.num_processes = num_processes
        self.num_steps = num_steps
        self.num_episodes = num_episodes
        self.max_steps = max_steps

        # 同時実行する環境分envを生成
        # 全エージェントが共有するBrainを生成
        self.envs = envs
        self.agent_processes = agents
        self.global_brain = global_brain

        self.rollouts = RolloutStorage(
            self.num_steps, self.num_processes
        )

    @classmethod
    def build(cls, builder):
        return A2CEnvironment(
            [builder.build_agent() for _ in range(builder.args.get('num_processes', 16))],
            [copy.copy(builder.test) for _ in range(builder.args.get('num_processes', 16))],
            global_brain=builder.build_brain(),
            num_processes=builder.args.get('num_processes', 16),
            num_steps=builder.args.get('num_steps', 5),
            num_episodes=builder.args.get('num_episodes', 300),
            max_steps=builder.args.get('max_steps', 200)
        )

    def run(self):
        """メイン"""
        episode_rewards = np.zeros(shape=[self.num_processes, 1])
        final_rewards = np.zeros(shape=[self.num_processes, 1])
        current_obs = np.array([self.envs[i].reset() for i in range(self.num_processes)])
        episode = 0

        self.rollouts.state[0] = current_obs.copy()

        for j in range(NUM_EPISODES * self.num_processes):

            for step in range(self.num_steps):
                # 行動を求める
                action = self.global_brain.get_actioin(self.rollouts.state[step])

                # (16,1) -> (16,) -> tensor をnumpyに
                actions = action.reshape((self.num_processes,))

                for i, agent in enumerate(self.agent_processes):
                    agent.state, agent.reward, agent.done, _ = self.envs[i].step(actions[i], agent.step)

                    if agent.done:
                        # 環境０の時のみ出力
                        if i == 0:
                            print(
                                '{} Episode: Finished after {} steps, total reward = {}'.format(episode, agent.step + 1,
                                                                                                self.envs[
                                                                                                    i].total_reward))
                            episode += 1

                        agent.step = 0  # stepのリセット
                        agent.state = self.envs[i].reset()

                    else:
                        agent.step += 1

                reward = [[agent.reward] for agent in self.agent_processes]
                episode_rewards += reward

                masks = np.array([[0.0] if ag.done else [1.0] for ag in self.agent_processes])

                # 最後の試行のそう報酬を更新
                final_rewards *= masks  # 継続中の場合は１を掛け算してそのまま、doneの時には０を掛け算してリセット

                # 継続中は０をたす、doneはepisode_rewardsをたす
                final_rewards += (1 - masks) * episode_rewards

                # 試行の総報酬を更新
                episode_rewards *= masks  # 継続中のmaskは１、doneの場合は０

                # 現在の状態をdoneのときは全部0
                current_obs *= masks

                # current_obsお更新
                current_obs = np.array([agent.state for agent in self.agent_processes])

                # メモリオブジェクトに今stepのtransitionを挿入
                self.rollouts.insert(current_obs, action, reward, masks)

            next_value = self.global_brain.get_value(self.rollouts.state[-1])

            self.rollouts.compute_return(next_value)

            self.global_brain.update(self.rollouts)

            self.rollouts.after_update()

            if self.envs[0].is_finish():
                break


STATE_SIZE = 8


def test():
    from logging import basicConfig, INFO, ERROR, CRITICAL, DEBUG
    basicConfig(level=ERROR)

    from tengu.oanda_action.oanda_dataframe import oanda_dataframe
    from tengu.drlfx.test_oanda import TestOanda

    df_org = oanda_dataframe('../../../../USD_JPY_M1.csv')

    test = TestOanda(df_org['close'].values, (600), STATE_SIZE)
    from tengu.drlfx.base_rl.nnet_builder.nnet_builder import NNetBuilder, BuilderArgument

    env = NNetBuilder(test, "A2C", args=BuilderArgument(),
                      environment=A2CEnvironment, agent=A2CAgent, brain=A2CBrain, nnet=A2CNet).build_environment()
    env.run()


if __name__ == '__main__':
    v = np.random.multinomial(10000, [0.1, 0.3, 0.1, 0.5])
    print(v)
    test()
