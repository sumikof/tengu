from dl.ddqn.agent import AgentDDQN
import numpy as np

NUM_EPISODES = 300
MAX_STEPS = 200



class EnvironmentDDQN:
    def __init__(self,env):

        self.env = env

        self.agent = AgentDDQN(self.env.num_status, self.env.num_actions)
        self.episode = 0
        self.step = 0

    def conv_tensor(self, state, shape):
        return np.reshape(state,shape)

    def run(self):
        complete_episodes = 0
        is_episode_final = False

        # 試行回数分実行
        for self.episode in range(NUM_EPISODES):

            print("start episode : "+str(self.episode))

            # 環境の初期化
            observation = self.env.reset()

            state = observation
            state = self.conv_tensor(state,[1,self.env.num_status])

            for self.step in range(MAX_STEPS):
                # main networkから行動を決定
                action = self.agent.get_action(state, self.episode)

                # 行動の実行（次の状態と終了判定を取得）
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.conv_tensor(next_state,[1,self.env.num_status])

                if done:
                    print("done is step : " + str(self.step))
                    next_state = np.zeros(state.shape)

                    # 報酬の設定
                    if self.step < 195:
                        # 失敗
                        reward = -1
                        complete_episodes = 0
                    else:
                        # 成功
                        reward = 1
                        complete_episodes += 1
                else:
                    # 終了時以外は報酬なし
                    reward = 0


                # メモリに追加
                self.agent.memorize(state, action, next_state, reward)

                # main q networkの更新
                self.agent.update_Q_function()

                state = next_state

                if done:
                    if self.episode % 2 == 0:
                        self.agent.update_target_Q_function()
                    break

            if is_episode_final is True:
                break

            # 10連続で達成できたら成功
            if complete_episodes > 10:
                print("成功")
                print("episode: "+str(self.episode))
                is_episode_final = True


if __name__ == '__main__':
    pass