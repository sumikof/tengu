class EnvironmentDDQN:
    def __init__(self, task, agent, num_episodes=300, max_steps=200):
        self.task = task
        self.agent = agent
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.step = 0

    def run(self):

        # 試行回数分実行
        for self.episode in range(self.num_episodes):

            print("start episode : " + str(self.episode))

            # 環境の初期化
            state = self.task.reset()

            done = False
            self.step = 0
            while not done:
                if self.max_steps != 0 and self.step > self.max_steps:
                    break

                # main networkから行動を決定
                action = self.agent.get_action(state, self.episode, self.task.mask)

                # 行動の実行（次の状態と終了判定を取得）
                next_state, reward, done, _ = self.task.step(action, self)

                # メモリに追加
                self.agent.memorize(state, action, next_state, reward)

                # main q networkの更新
                self.agent.update_Q_function()

                state = next_state

                if done:
                    if self.episode % 2 == 0:
                        self.agent.update_target_Q_function()
                    break

                self.step += 1

            if self.task.is_finish():
                print("成功")
                print("episode: " + str(self.episode))
                break


if __name__ == '__main__':
    pass
