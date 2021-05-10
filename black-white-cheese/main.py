# 导入黑白棋文件
from game import Game
# 导入随机包
import random

import math
import copy


class HumanPlayer:
    """
    人类玩家
    """

    def __init__(self, color):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """
        self.color = color

    def get_move(self, board):
        """
        根据当前棋盘输入人类合法落子位置
        :param board: 棋盘
        :return: 人类下棋落子位置
        """
        # 如果 self.color 是黑棋 "X",则 player 是 "黑棋"，否则是 "白棋"
        if self.color == "X":
            player = "黑棋"
        else:
            player = "白棋"

        # 人类玩家输入落子位置，如果输入 'Q', 则返回 'Q'并结束比赛。
        # 如果人类玩家输入棋盘位置，e.g. 'A1'，
        # 首先判断输入是否正确，然后再判断是否符合黑白棋规则的落子位置
        while True:
            action = input(
                "请'{}-{}'方输入一个合法的坐标(e.g. 'D3'，若不想进行，请务必输入'Q'结束游戏。): ".format(player,
                                                                             self.color))

            # 如果人类玩家输入 Q 则表示想结束比赛
            if action == "Q" or action == 'q':
                return "Q"
            else:
                row, col = action[1].upper(), action[0].upper()

                # 检查人类输入是否正确
                if row in '12345678' and col in 'ABCDEFGH':
                    # 检查人类输入是否为符合规则的可落子位置
                    if action in board.get_legal_actions(self.color):
                        return action
                else:
                    print("你的输入不合法，请重新输入!")


class RandomPlayer:
    """
    随机玩家, 随机返回一个合法落子位置
    """

    def __init__(self, color):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """
        self.color = color

    def random_choice(self, board):
        """
        从合法落子位置中随机选一个落子位置
        :param board: 棋盘
        :return: 随机合法落子位置, e.g. 'A1'
        """
        # 用 list() 方法获取所有合法落子位置坐标列表
        action_list = list(board.get_legal_actions(self.color))

        # 如果 action_list 为空，则返回 None,否则从中选取一个随机元素，即合法的落子坐标
        if len(action_list) == 0:
            return None
        else:
            return random.choice(action_list)

    def get_move(self, board):
        """
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g. 'A1'
        """
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))
        action = self.random_choice(board)
        return action


class AIPlayer:
    """
    AI 玩家
    """
    step = 0
    def __init__(self, color):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """
        # 最大迭代次数
        self.max_times = 100
        # 玩家颜色
        self.color = color
        # UCB超参数
        self.SCALAR = 1

    def get_move(self, board):
        """
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g. 'A1'
        """
        print(self.step)
        self.step = self.step + 1
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))

        # -----------------请实现你的算法代码--------------------------------------
        board_state = copy.deepcopy(board)
        root = Node(state=board_state, color=self.color)
        root.visits = 1
        action = self.uct_search(self.max_times, root)
        # ------------------------------------------------------------------------

        return action

    def uct_search(self, max_times, root):
        """
        根据当前棋盘状态获取最佳落子位置
        :param max_times: 最大搜索次数，默认100
        :param root: 根节点
        :return: action 最佳落子位置, e.g. 'A1'
        """
        for t in range(max_times):
            # print(t)
            leave = self.select_policy(root)
            reward = self.stimulate_policy(leave)
            self.backup(leave, reward)
            best_child = self.ucb(root, 0)
        return best_child.action

    def select_policy(self, node):
        """
        选择扩展的节点
        :param node: 根节点，Node 类
        :return: leave:Node 类
        """
        while not self.terminal(node.state):
            #node.state.display()
            #print(list(node.state.get_legal_actions(node.color)))
            if len(node.children) == 0:
                new_node = self.expand(node)
                #print(new_node.action)
                return new_node
            elif random.uniform(0, 1) < .5:
                node = self.ucb(node, self.SCALAR)
            else:
                node = self.ucb(node, self.SCALAR)
                if not node.fully_expanded():
                    return self.expand(node)
                else:
                    node = self.ucb(node, self.SCALAR)
        return node

    # 扩展函数
    def expand(self, node):
        """
        选择扩展的节点
        :param node: 根节点，Node 类
        :return: leave:Node 类
        """
        # 随机选择动作
        action_list = list(node.state.get_legal_actions(node.color))
        # 防止尾盘时出现卡死，没有动作可以选择
        if len(action_list) == 0:
            return node.parent

        action = random.choice(action_list)
        tried_action = [c.action for c in node.children]
        while action in tried_action:
            action = random.choice(action_list)

        # 复制状态并根据动作更新到新状态
        new_state = copy.deepcopy(node.state)
        new_state._move(action, node.color)

        # 确定子节点颜色
        if node.color == 'X':
            new_color = 'O'
        else:
            new_color = 'X'

        # 新建节点
        node.add_child(new_state, action=action, color=new_color)
        return node.children[-1]

    # ucb选择函数
    def ucb(self, node, scalar):
        """
        选择最佳子节点
        :param node: 节点，Node 类
        :param scalar: UCT公式超参数
        :return: best_child:最佳子节点，Node 类
        """
        best_score = -float('inf')
        best_children = []
        for c in node.children:
            exploit = c.reward / c.visits
            if c.visits == 0:
                best_children = [c]
                break
            explore = math.sqrt(2.0 * math.log(node.visits) / float(c.visits))
            score = exploit + scalar * explore
            if score == best_score:
                best_children.append(c)
            if score > best_score:
                best_children = [c]
                best_score = score
        if len(best_children) == 0:
            return node.parent
        return random.choice(best_children)

    def stimulate_policy(self, node):
        """
        随机模拟对弈
        :param node: 节点，Node 类
        :return: reward:期望值
        在定义期望值时同时考虑了胜负关系和获胜的子数，board.get_winner()会返回胜负关系和获胜子数
        在这里我们定义获胜积100分，每多赢一个棋子多1分
        reward = 100 + difference
        """
        board = copy.deepcopy(node.state)
        color = node.color
        count = 0
        while not self.terminal(board):
            action_list = list(node.state.get_legal_actions(color))
            if not len(action_list) == 0:
                action = random.choice(action_list)
                board._move(action, color)
                if color == 'X':
                    color = 'O'
                else:
                    color = 'X'
            else:
                if color == 'X':
                    color = 'O'
                else:
                    color = 'X'
                action_list = list(node.state.get_legal_actions(color))
                action = random.choice(action_list)
                board._move(action, color)
                if color == 'X':
                    color = 'O'
                else:
                    color = 'X'
            count = count + 1
            if count >= 10:
                break

        # 价值函数定义
        winner, difference = board.get_winner()
        if winner == 2:
            reward = 0
        elif winner == 1:
            reward = 100 + difference
        else:
            reward = -(100 + difference)

        if self.color == 'X':
            reward = - reward
        return reward

    def backup(self, node, reward):
        while node.parent is not None:
            node.visits += 1
            if node.parent.color == self.color:
                node.reward += reward
            else:
                node.reward -= reward
            node = node.parent
        return 0

    def terminal(self, state):
        """
        判断游戏是否结束
        :return: True/False 游戏结束/游戏没有结束
        """

        # 根据当前棋盘，判断棋局是否终止
        # 如果当前选手没有合法下棋的位子，则切换选手；如果另外一个选手也没有合法的下棋位置，则比赛停止。
        b_list = list(state.get_legal_actions('X'))
        w_list = list(state.get_legal_actions('O'))

        is_over = len(b_list) == 0 and len(w_list) == 0  # 返回值 True/False

        return is_over


class Node:
    def __init__(self, state, parent=None, action=None, color="X"):
        self.visits = 0  #访问次数
        self.reward = 0.0 #期望值
        self.state = state #棋盘状态，Broad类
        self.children = [] #子节点
        self.parent = parent #父节点
        self.action = action #从父节点转移到本节点采取的动作
        self.color = color #该节点玩家颜色

    # 增加子节点
    def add_child(self, child_state, action, color):
        child_node = Node(child_state, parent=self, action=action, color=color)
        self.children.append(child_node)

    # 判断是否完全展开
    def fully_expanded(self):
        action = list(self.state.get_legal_actions(self.color))
        if len(self.children) == len(action):
            return True
        return False


# 人类玩家黑棋初始化
black_player = RandomPlayer("X")

# AI 玩家 白棋初始化
white_player = AIPlayer("O")

# AI 玩家 白棋初始化
# white_player = HumanPlayer("O")

# 游戏初始化，第一个玩家是黑棋，第二个玩家是白棋
game = Game(black_player, white_player)

# 开始下棋
game.run()
