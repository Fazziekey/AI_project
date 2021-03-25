# Developer：Fazzie
# Time: 2021/3/2116:23
# File name: queen.py
# Development environment: Anaconda Python
import numpy as np  # 提供维度数组与矩阵运算
import copy  # 从copy模块导入深度拷贝方法
from board import Chessboard

'''
# 初始化8*8八皇后棋盘
chessboard = Chessboard()

# 在棋盘上的坐标点（4，4）落子
chessboard.setQueen(4,4)

# 方法一，逐子落子
# 选择False不打印中间过程棋盘
# 完成八皇后落子
# 终局胜负条件判定及输出

chessboard.boardInit(False)
chessboard.setQueen(0,0,False)
chessboard.setQueen(1,6,False)
chessboard.setQueen(2,4,False)
chessboard.setQueen(3,7,False)
chessboard.setQueen(4,1,False)
chessboard.setQueen(5,3,False)
chessboard.setQueen(6,5,False)
chessboard.setQueen(7,2,False)
chessboard.printChessboard(False)
print("Win?    ----    ",chessboard.isWin())

# 方法二，序列落子
# 选择False不打印中间过程棋盘
# 完成八皇后落子
# 终局胜负条件判定及输出
chessboard.boardInit(False)
Queen_setRow = [0,6,4,7,1,3,5,2]
for i,item in enumerate(Queen_setRow):
    chessboard.setQueen(i,item,False)
chessboard.printChessboard(False)
print("Win?    ----    ",chessboard.isWin())


# 开放接口
# 让玩家自行体验八皇后游戏
chessboard = Chessboard()
chessboard.play()

'''


# 基于棋盘类，设计搜索策略
class Game:
    def __init__(self, show=True):
        """
        初始化游戏状态.
        """

        self.chessBoard = Chessboard(show)
        self.solves = []
        self.solve = []
        self.gameInit()

    # 重置游戏
    def gameInit(self, show=True):
        """
        重置棋盘.
        """

        self.Queen_setRow = [-1] * 8
        self.chessBoard.boardInit(False)

    ##############################################################################
    ####                请在以下区域中作答(可自由添加自定义函数)                 ####
    ####              输出：self.solves = 八皇后所有序列解的list                ####
    ####             如:[[0,6,4,7,1,3,5,2],]代表八皇后的一个解为                ####
    ####           (0,0),(1,6),(2,4),(3,7),(4,1),(5,3),(6,5),(7,2)            ####
    ##############################################################################
    #                                                                            #

    def run(self, row=0):
        if row == 8:
            self.solves.append(list(self.solve))
            #print("---/n", self.solves)
        for column in range(8):
            if self.isvalid(column):
                self.solve.append(column)
                # print(self.solve)
                self.run(row+1)
                self.solve.pop()

    #                                                                            #
    ##############################################################################
    #################             完成后请记得提交作业             #################
    ##############################################################################

    def isvalid(self,column):
        for i in range(len(self.solve)):
            if (len(self.solve) - i) == abs(column - self.solve[i]) or self.solve[i] == column:
                return False
        return True

    def showResults(self, result):
        """
        结果展示.
        """

        self.chessBoard.boardInit(False)
        for i, item in enumerate(result):
            if item >= 0:
                self.chessBoard.setQueen(i, item, False)

        self.chessBoard.printChessboard(False)

    def get_results(self):
        """
        输出结果(请勿修改此函数).
        return: 八皇后的序列解的list.
        """

        self.run()
        print("---/n", self.solves)
        return self.solves


game = Game()
solutions = game.get_results()
print('There are {} results.'.format(len(solutions)))
#print(len(solutions[0]))
# print(solutions)
game.showResults(solutions[0])
