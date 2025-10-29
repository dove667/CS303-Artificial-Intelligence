import numpy as np
import random
import time
from typing import List, Tuple, Set, Optional

COLOR_BLACK=-1
COLOR_WHITE=1
COLOR_NONE=0
INFINITY = float('inf')
random.seed(0)

#don’t change the class name
class AI(object):
    #chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color , time_out):
        self.chessboard_size = chessboard_size
        # You are white or black
        self.color = color
        # the max time you should use, your algorithm’s run
        # time must not exceed the time limit.
        self.time_out = time_out
        # You need to add your decision to your candidate_list.
        # The system will get the end of your candidate_list as your decision.
        self.candidate_list = []
        self.directions = [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1),          (0, 1),
                           (1, -1),  (1, 0), (1, 1)]
        self.max_depth = 6
    # The input is the current chessboard. Chessboard is a numpy array.
    def _is_valid_move(self, chessboard, row, col, color) -> Tuple[bool, Optional[List[Tuple[int, int]]]]:
        """
        判断一个落子是否合法，如果是合法的，返回True和需要翻转的棋子位置列表，否则返回False和None。
        Args:
            chessboard: 当前棋盘状态
            row: 落子行号
            col: 落子列号
            color: 落子颜色
        Returns:
            Tuple[bool, Optional[List[Tuple[int, int]]]]: 是否合法及需要翻转的棋子位置列表
        """
        if chessboard[row, col] != COLOR_NONE:
            return False, None
   
        flips_all = []
        for dr, dc in self.directions:
            r, c = row + dr, col + dc
            line = []
            while 0 <= r < self.chessboard_size and 0 <= c < self.chessboard_size and chessboard[r, c] == -color:
                line.append((r, c))
                r += dr
                c += dc
            if 0 <= r < self.chessboard_size and 0 <= c < self.chessboard_size and chessboard[r, c] == color and line:
                flips_all.extend(line)

        if flips_all:
            return True, flips_all
        return False, None

    def _get_possible_moves(self, chessboard, color) -> Set[Tuple[int, int]]:
        occupied = np.where(chessboard == -color) # 返回(array of row indices, array of col indices)
        possible_moves = set() 
        for pos in zip(occupied[0], occupied[1]):
            for dr, dc in self.directions:
                new_row, new_col = pos[0] + dr, pos[1] + dc
                if 0 <= new_row < self.chessboard_size and 0 <= new_col < self.chessboard_size and chessboard[new_row, new_col] == COLOR_NONE:
                    possible_moves.add((new_row, new_col))
        return possible_moves

    def get_candidate_reversed_list(self, chessboard, color) -> Tuple[List[Tuple[int, int]], List[List[Tuple[int, int]]]]:
        """
        获取所有合法落子点列表和对应的翻转棋子位置列表
        """
        possible_moves = self._get_possible_moves(chessboard, color)
        candidate_list = []
        reversed_list = []
        for pos in possible_moves:
            is_valid, reversed_opponent = self._is_valid_move(chessboard, pos[0], pos[1], color)
            if is_valid and reversed_opponent:
                candidate_list.append(pos)
                reversed_list.append(reversed_opponent)
        return candidate_list, reversed_list
    
    def is_terminal(self, chessboard) -> bool:
        """双方都无合法步则终局"""
        cand_b, _ = self.get_candidate_reversed_list(chessboard, COLOR_BLACK)
        if cand_b:
            return False
        cand_w, _ = self.get_candidate_reversed_list(chessboard, COLOR_WHITE)
        return len(cand_w) == 0
        
    def _get_stable_disk(self, chessboard) -> int:
        """
        计算颜色的稳定子得分
        但是目前只是一个不准确的估计。
        """
        stable_coords = set()
        corners = [(0, 0), (0, self.chessboard_size - 1),
                (self.chessboard_size - 1, 0), (self.chessboard_size - 1, self.chessboard_size - 1)]
        corner_dirs = {
            (0, 0): [(0, 1), (1, 0), (1, 1)],
            (0, self.chessboard_size - 1): [(0, -1), (1, 0), (1, -1)],
            (self.chessboard_size - 1, 0): [(0, 1), (-1, 0), (-1, 1)],
            (self.chessboard_size - 1, self.chessboard_size - 1): [(0, -1), (-1, 0), (-1, -1)],
        }
        for cr, cc in corners:
            if chessboard[cr, cc] == COLOR_NONE:
                continue
            color = chessboard[cr, cc]
            for dr, dc in corner_dirs[(cr, cc)]:
                r, c = cr, cc
                while 0 <= r < self.chessboard_size and 0 <= c < self.chessboard_size and chessboard[r, c] == color:
                    stable_coords.add((r, c))
                    r += dr
                    c += dc
        return sum(int(chessboard[r, c]) for (r, c) in stable_coords)

    def _get_weights(self, piece_count):
        """
        根据当前回合数返回不同阶段的权重设置
        returns: (w1, w2, w3，w4)
        其中w1是棋盘位置权重，w2是确定子数量，w3是翻转棋子个数，w4是行动力
        """
        if piece_count < 15: # 开局,确定子重要
            return (0.3, 0.5, 0.1, 0.1)
        elif piece_count < 45: # 中局，行动力重要
            return (0.3, 0.2, 0.2, 0.3)
        else: # 残局，翻转子重要
            return (0.3, 0.1, 0.5, 0.1)
        
    def evaluate(self, chessboard, piece_count) -> float:
        """
        反黑白棋启发式评估（“白方优势”为正，“黑方优势”为负）。
        特征：
        - 位置权重：weighted_chessboard
        - 稳定子估计：stable_score（角出发的保守估计）
        - 行动力：mobility = 白可行步数 - 黑可行步数
        说明：
        - 不再使用“上一手翻子数”等与当前局面不稳定、符号易错的特征。
        - 对于反黑白棋，保持“白正黑负”的视角，并在极大极小中令黑方max、白方min，可自洽。
        """
        weighted_board = np.array([
            [1, 8, 3, 7, 7, 3, 8, 1],
            [8, 3, 2, 5, 5, 2, 3, 8],
            [3, 2, 6, 6, 6, 6, 2, 3],
            [7, 5, 6, 4, 4, 6, 5, 7],
            [7, 5, 6, 4, 4, 6, 5, 7],
            [3, 2, 6, 6, 6, 6, 2, 3],
            [8, 3, 2, 5, 5, 2, 3, 8],
            [1, 8, 3, 7, 7, 3, 8, 1],
        ])
        # 位置权重（白正黑负）
        weighted_chessboard = float(np.sum(weighted_board * chessboard))
    
        # 稳定子估计（白正黑负）
        stable_score = float(self._get_stable_disk(chessboard))

        # 行动力（白可行步数 - 黑可行步数）
        white_moves, _ = self.get_candidate_reversed_list(chessboard, COLOR_WHITE)
        black_moves, _ = self.get_candidate_reversed_list(chessboard, COLOR_BLACK)
        mobility = float(len(white_moves) - len(black_moves))

        # 分阶段权重
        w1, w2, w3, w4 = self._get_weights(piece_count)
        score = w1 * weighted_chessboard + w2 * stable_score + w3 * 0.0 + w4 * mobility
        return score

    def minimax_search(self, chessboard, piece_count) -> Tuple[float, Tuple[int, int]]:
        """
        黑棋（-1）为max节点，白棋（+1）为min节点。
        评估函数以“白方优势”为正数，故该极大极小方向在反黑白棋下自洽。
        Args:
            chessboard: 当前棋盘状态
            candidate_list: 当前可选的落子点列表
        Returns:
            Tuple[float, Tuple[int, int]]: 最佳评估值及对应的落子点
        """
        def max_value(chessboard, depth, alpha, beta, piece_count) -> Tuple[float, Tuple[int, int]]:
            if depth == self.max_depth or self.is_terminal(chessboard):
                return self.evaluate(chessboard, piece_count), None

            candidate_list, reversed_list = self.get_candidate_reversed_list(chessboard, COLOR_BLACK)
            if not candidate_list:
                return min_value(chessboard, depth + 1, alpha, beta, piece_count)

            best, move = -INFINITY, None
            for candidate, reversed_opponent in zip(candidate_list, reversed_list):
                # 执行落子
                chessboard[candidate[0], candidate[1]] = COLOR_BLACK
                for r, c in reversed_opponent:
                    chessboard[r, c] = COLOR_BLACK

                v2, _ = min_value(chessboard, depth + 1, alpha, beta, piece_count + 1)
                if v2 > best:
                    best, move = v2, candidate

                # 回退
                for r, c in reversed_opponent:
                    chessboard[r, c] = COLOR_WHITE
                chessboard[candidate[0], candidate[1]] = COLOR_NONE

                # alpha-beta
                if best > alpha:
                    alpha = best
                if alpha >= beta:
                    break
            return best, move

        def min_value(chessboard, depth, alpha, beta, piece_count) -> Tuple[float, Tuple[int, int]]:
            if depth == self.max_depth or self.is_terminal(chessboard):
                return self.evaluate(chessboard, piece_count), None

            candidate_list, reversed_list = self.get_candidate_reversed_list(chessboard, COLOR_WHITE)
            if not candidate_list:
                return max_value(chessboard, depth + 1, alpha, beta, piece_count)

            best, move = INFINITY, None
            for candidate, reversed_opponent in zip(candidate_list, reversed_list):
                # 执行落子
                chessboard[candidate[0], candidate[1]] = COLOR_WHITE
                for r, c in reversed_opponent:
                    chessboard[r, c] = COLOR_WHITE

                v2, _ = max_value(chessboard, depth + 1, alpha, beta, piece_count + 1)
                if v2 < best:
                    best, move = v2, candidate

                # 回退
                for r, c in reversed_opponent:
                    chessboard[r, c] = COLOR_BLACK
                chessboard[candidate[0], candidate[1]] = COLOR_NONE

                # alpha-beta
                if best < beta:
                    beta = best
                if alpha >= beta:
                    break
            return best, move

        if self.color == COLOR_BLACK:
            return max_value(chessboard, 0, -INFINITY, INFINITY, piece_count) # 黑棋希望评估值越大越好
        else:
            return min_value(chessboard, 0, -INFINITY, INFINITY, piece_count) # 白棋希望评估值越小越好

    def go(self, chessboard):
        self.candidate_list.clear()
        #============================================
        #Write your algorithm here
        # 确认所有合法落子点
        self.candidate_list, _ = self.get_candidate_reversed_list(chessboard, self.color)
        # 选择最终落子点
        if not self.candidate_list:         
            return self.candidate_list
        piece_count = np.sum(chessboard != COLOR_NONE)
        _, final_move = self.minimax_search(chessboard, piece_count)
        if final_move is not None:
            self.candidate_list.append(final_move)
        return self.candidate_list
       