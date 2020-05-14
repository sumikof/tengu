from dl.ddqn.agent import AgentDDQN
import numpy as np

NUM_EPISODES = 300
MAX_STEPS = 200



class EnvironmentDDQN:
    def __init__(self,env):
        self.env = env
        self.agent = AgentDDQN(self.env.num_status, self.env.num_actions)

    def conv_tensor(self, state, shape):
        return np.reshape(state,shape)

    def run(self):

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
                next_state, reward, done, _ = self.env.step(action,self)

                # メモリに追加
                self.agent.memorize(state, action, next_state, reward)

                # main q networkの更新
                self.agent.update_Q_function()

                state = next_state

                if done:
                    if self.episode % 2 == 0:
                        self.agent.update_target_Q_function()
                    break


            if self.env.is_finish():
                print("成功")
                print("episode: "+str(self.episode))
                break


if __name__ == '__main__':
    pass