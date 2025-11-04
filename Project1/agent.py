import numpy as np
import random
import time
from typing import List, Tuple, Optional
from numba import njit


COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
INFINITY = float('inf')
random.seed(0)

DIRS = np.array([
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),          (0, 1),
    (1, -1), (1, 0), (1, 1)
], dtype=np.int8)

@njit
def nb_get_possible_moves(board, color):
    """
    返回所有“与对手棋子相邻且为空”的点，形状 (K,2) 的 int32 数组。
    """
    opp = (board == -color)
    neighbor = np.zeros_like(opp)
    # 上下左右
    neighbor[:-1, :] |= opp[1:, :]
    neighbor[1:,  :] |= opp[:-1, :]
    neighbor[:, :-1] |= opp[:, 1:]
    neighbor[:, 1: ] |= opp[:, :-1]
    # 对角
    neighbor[:-1, :-1] |= opp[1:, 1:]
    neighbor[:-1, 1: ] |= opp[1:, :-1]
    neighbor[1:,  :-1] |= opp[:-1, 1:]
    neighbor[1:,  1: ] |= opp[:-1, :-1]

    possible = neighbor & (board == 0)
    rows, cols = np.nonzero(possible)
    k = rows.shape[0]
    coords = np.empty((k, 2), dtype=np.int32)
    coords[:, 0] = rows
    coords[:, 1] = cols
    return coords

@njit
def nb_is_valid_move(board, row, col, color):
    """
    判断合法性并返回需要翻转的坐标数组 flips，形状 (M,2)。
    无翻子则返回 (False, 空数组)。
    """
    n = board.shape[0]
    if board[row, col] != 0:
        return False, np.empty((0, 2), dtype=np.int32)

    flips = np.empty((64, 2), dtype=np.int32)  
    idx = 0
    for dr, dc in DIRS:
        r = row + dr; c = col + dc
        # 第一格必须是对手棋子
        if r < 0 or r >= n or c < 0 or c >= n or board[r, c] != -color:
            continue
        cnt = 0
        while 0 <= r < n and 0 <= c < n and board[r, c] == -color:
            cnt += 1
            r += dr; c += dc
        if cnt > 0 and 0 <= r < n and 0 <= c < n and board[r, c] == color:
            # 回填需要翻转的格子
            for t in range(1, cnt + 1):
                rr = row + dr * t
                cc = col + dc * t
                flips[idx, 0] = rr
                flips[idx, 1] = cc
                idx += 1
    if idx == 0:
        return False, np.empty((0, 2), dtype=np.int32)
    return True, flips[:idx]

#don’t change the class name
class AI(object):
    #chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color , time_out=4.5):
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
        self.max_depth = 64
        self.weighted_board = np.array([
            [1, 8, 3, 7, 7, 3, 8, 1],
            [8, 3, 2, 5, 5, 2, 3, 8],
            [3, 2, 6, 6, 6, 6, 2, 3],
            [7, 5, 6, 4, 4, 6, 5, 7],
            [7, 5, 6, 4, 4, 6, 5, 7],
            [3, 2, 6, 6, 6, 6, 2, 3],
            [8, 3, 2, 5, 5, 2, 3, 8],
            [1, 8, 3, 7, 7, 3, 8, 1],
        ], dtype=np.int16)
        # 用于各层走法排序的主变线提示：pv_moves[ply] = 该层推荐走法(根为0层)
        self.pv_moves: List[Optional[Tuple[int,int]]] = [None] * (self.max_depth)

    def _is_valid_move(self, chessboard, row, col, color):
        """
        判断一个落子是否合法，如果是合法的，返回True和需要翻转的棋子位置列表，否则返回False和None。
        """
        return nb_is_valid_move(chessboard, int(row), int(col), int(color))

    def _get_possible_moves(self, chessboard, color):
        """
        返回形状 (K,2) 的 ndarray，更适合数值计算；遍历可直接 for r,c in arr
        """
        return nb_get_possible_moves(chessboard, int(color))
    
    def get_candidate_reversed_list(self, chessboard, color) -> Tuple[List[Tuple[int, int]], List[List[Tuple[int, int]]]]:
        """
        获取所有合法落子点列表和对应的翻转棋子位置列表
        """
        possible_moves = self._get_possible_moves(chessboard, color)
        candidate_list = []
        reversed_list = []
        for r, c in possible_moves:
            ok, flips = self._is_valid_move(chessboard, r, c, color)  # flips: (M,2) ndarray
            if ok and flips.shape[0] > 0:
                candidate_list.append((int(r), int(c)))
                reversed_list.append(flips)
        return candidate_list, reversed_list
    
    def is_terminal(self, chessboard) -> bool:
        """双方都无合法步则终局"""
        return (not self._has_any_valid_move(chessboard, COLOR_BLACK)
                and not self._has_any_valid_move(chessboard, COLOR_WHITE))
    
    def _has_any_valid_move(self, chessboard, color) -> bool:
        """只要发现一个合法步就返回 True，减少不必要计算"""
        for r, c in self._get_possible_moves(chessboard, color):
            ok, flips = self._is_valid_move(chessboard, r, c, color)
            if ok and flips.shape[0] > 0:
                return True
        return False
    
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
        
    def evaluate(self, chessboard) -> float:
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
        # 位置权重（白正黑负）
        weighted_chessboard = float(np.sum(self.weighted_board * chessboard))

        # 稳定子估计（白正黑负）
        stable_score = float(self._get_stable_disk(chessboard))

        # 行动力（白可行步数 - 黑可行步数）
        white_moves, _ = self.get_candidate_reversed_list(chessboard, COLOR_WHITE)
        black_moves, _ = self.get_candidate_reversed_list(chessboard, COLOR_BLACK)
        mobility = float(len(white_moves) - len(black_moves))

        # 分阶段权重
        piece_count = int(np.sum(chessboard != COLOR_NONE))
        w1, w2, w3, w4 = self._get_weights(piece_count)
        score = w1 * weighted_chessboard + w2 * stable_score + w3 * 0.0 + w4 * mobility
        return score

    def minimax_search(self, chessboard, depth_limit, deadline) -> Tuple[float, Tuple[int, int], bool, List[Tuple[int,int]]]:
        """
        黑棋（-1）为max节点，白棋（+1）为min节点。
        评估函数以“白方优势”为正数，故该极大极小方向在反黑白棋下自洽。
        Args:
            chessboard: 当前棋盘状态
            candidate_list: 当前可选的落子点列表
        Returns:
            Tuple[float, Tuple[int, int], bool, List[Tuple[int,int]]]: (score, move, timed_out, pv_moves)
        """
        def time_exceeded() -> bool:
            return time.time() > deadline

        def reorder_with_pv(cands, revs, depth):
            # 在每一层用上一轮的主变线置顶
            mv = self.pv_moves[depth] if depth < len(self.pv_moves) else None
            if mv is None or not cands:
                return
            try:
                i = cands.index(mv)
                if i != 0:
                    cands[0], cands[i] = cands[i], cands[0]
                    revs[0],  revs[i]  = revs[i],  revs[0]
            except ValueError:
                pass

        def max_value(chessboard, depth, alpha, beta) -> Tuple[float, Optional[Tuple[int, int]], bool, List[Tuple[int,int]]]:
            if time_exceeded():
                return self.evaluate(chessboard), None, True, []
            if depth == depth_limit or self.is_terminal(chessboard):
                return self.evaluate(chessboard), None, False, []

            candidate_list, reversed_list = self.get_candidate_reversed_list(chessboard, COLOR_BLACK)
            reorder_with_pv(candidate_list, reversed_list, depth)
            if not candidate_list:
                return min_value(chessboard, depth + 1, alpha, beta)

            best, move = -INFINITY, None
            best_pv: List[Tuple[int,int]] = []
            for candidate, reversed_opponent in zip(candidate_list, reversed_list):
                r0, c0 = candidate
                # 执行落子（向量化翻子）
                chessboard[r0, c0] = COLOR_BLACK
                k = reversed_opponent.shape[0]
                if k:
                    rr = reversed_opponent[:, 0]
                    cc = reversed_opponent[:, 1]
                    chessboard[rr, cc] = COLOR_BLACK

                v2, _, timed_out, child_pv = min_value(chessboard, depth + 1, alpha, beta)
                
                # 回退（向量化回退）
                if k:
                    chessboard[rr, cc] = COLOR_WHITE
                chessboard[r0, c0] = COLOR_NONE

                if timed_out:
                    return best, move, True, best_pv
                
                if v2 > best:
                    best, move = v2, candidate
                    best_pv = [candidate] + child_pv
                # alpha-beta
                if best > alpha:
                    alpha = best
                if alpha >= beta:
                    break
            return best, move, False, best_pv

        def min_value(chessboard, depth, alpha, beta) -> Tuple[float, Optional[Tuple[int, int]], bool, List[Tuple[int,int]]]:
            if time_exceeded():
                return self.evaluate(chessboard), None, True, []
            if depth == depth_limit or self.is_terminal(chessboard):
                return self.evaluate(chessboard), None, False, []

            candidate_list, reversed_list = self.get_candidate_reversed_list(chessboard, COLOR_WHITE)
            reorder_with_pv(candidate_list, reversed_list, depth)
            if not candidate_list:
                return max_value(chessboard, depth + 1, alpha, beta)

            best, move = INFINITY, None
            best_pv: List[Tuple[int,int]] = []
            for candidate, reversed_opponent in zip(candidate_list, reversed_list):
                r0, c0 = candidate
                # 执行落子（向量化翻子）
                chessboard[r0, c0] = COLOR_WHITE
                k = reversed_opponent.shape[0]
                if k:
                    rr = reversed_opponent[:, 0]
                    cc = reversed_opponent[:, 1]
                    chessboard[rr, cc] = COLOR_WHITE

                v2, _, time_out, child_pv = max_value(chessboard, depth + 1, alpha, beta)

                # 回退（向量化回退）
                if k:
                    chessboard[rr, cc] = COLOR_BLACK
                chessboard[r0, c0] = COLOR_NONE

                if time_out:
                    return best, move, True, best_pv
                
                if v2 < best:
                    best, move = v2, candidate
                    best_pv = [candidate] + child_pv    
                # alpha-beta
                if best < beta:
                    beta = best
                if alpha >= beta:
                    break
            return best, move, False, best_pv

        if self.color == COLOR_BLACK:
            return max_value(chessboard, 0, -INFINITY, INFINITY) # 黑棋希望评估值越大越好
        else:
            return min_value(chessboard, 0, -INFINITY, INFINITY) # 白棋希望评估值越小越好

    def iterative_deepening(self, chessboard, start_time) -> Tuple[Optional[Tuple[int,int]], Optional[float]]:
        """
        迭代加深，在超时或达到上限即停止。
        优先返回已完成的最新深度结果。
        """
        best_move = None
        best_score = None
        self.pv_moves = [None] * (self.max_depth)
        deadline = start_time + min(self.time_out, 4.5)

        for depth in range(1, self.max_depth + 1):
            # IDDFS 外层超时检查（留余量）
            if time.time() > deadline:
                break
            score, move, timed_out, pv = self.minimax_search(chessboard, depth, deadline)
            if timed_out:
                # 当前深度未完成，停止并使用上一轮的结果
                break
            if move is not None:
                best_move = move
                best_score = score
                # 更新主变线提示，供下一轮各层排序使用
                self.pv_moves = [None] * (self.max_depth)
                for i, mv in enumerate(pv):
                    if i < len(self.pv_moves):
                        self.pv_moves[i] = mv
        return best_move, best_score
    
    def go(self, chessboard):
        start_time = time.time()
        self.candidate_list.clear()
        #============================================
        #Write your algorithm here
        # 确认所有合法落子点
        self.candidate_list, _ = self.get_candidate_reversed_list(chessboard, self.color)

        if not self.candidate_list:         
            return self.candidate_list
        # 选择最终落子点
        best_move, _ = self.iterative_deepening(chessboard, start_time)
        if best_move is not None:
            self.candidate_list.append(best_move)
        return self.candidate_list
       