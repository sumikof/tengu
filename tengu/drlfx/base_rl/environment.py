from logging import getLogger

logger = getLogger(__name__)


class EnvironmentDDQN:
    def __init__(self, agent, task, num_episodes, max_steps, train_interval, save_weight=False, weight_file_name=None):
        self.task = task
        self.agent = agent
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.step = 0
        self.train_interval = train_interval
        self.save_weights = save_weight
        self.weight_file_name = weight_file_name

    @classmethod
    def build(cls, builder):
        return EnvironmentDDQN(
            builder.build_agent(),
            builder.test,
            builder.args.get('num_episodes', 300),
            builder.args.get('max_steps', 200),
            builder.args.get('train_interval', 2),
            builder.args.get('save_weights'),
            builder.args.get('weight_file_name')
        )

    def run(self):

        # 試行回数分実行
        for self.episode in range(self.num_episodes):

            print("start episode : " + str(self.episode))

            # 環境の初期化
            state = self.task.reset()

            done = False
            self.step = 0
            while not done:
                logger.debug("step = {} start".format(self.step))
                if self.max_steps != 0 and self.step > self.max_steps:
                    break

                # main networkから行動を決定
                action = self.agent.get_action(state, self.episode, self.task.mask)
                logger.debug("get action {}".format(action))

                # 行動の実行（次の状態と終了判定を取得）
                next_state, reward, done, _ = self.task.step(action, step=self.step)
                logger.debug("task.step done={}".format(done))

                # メモリに追加
                self.agent.memorize(state, action, next_state, reward)

                # main q networkの更新
                if self.step % self.train_interval == 0:
                    self.agent.update_Q_function()

                state = next_state

                if done:
                    if self.episode % 2 == 0:
                        self.agent.update_target_Q_function()

                    if self.save_weights:
                        self.agent.save_weights(self.weight_file_name)

                    break

                self.step += 1

            if self.task.is_finish():
                print("成功")
                print("episode: " + str(self.episode))
                break


if __name__ == '__main__':
    pass
