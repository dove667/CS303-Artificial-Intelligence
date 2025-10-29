from collections import deque

def bfs(graph, start_state, goal_state):
    explored = set()                     # 已访问节点
    frontier = deque([start_state])      # 待探索队列
    parent = {start_state: None}         # 记录每个节点的父节点

    while frontier:
        state = frontier.popleft()
        explored.add(state)

        # 找到目标节点 -> 回溯路径
        if state == goal_state:
            path = []
            while state is not None:
                path.append(state)
                state = parent[state]
            return list(reversed(path))  # 反转得到从起点到终点的路径

        # 遍历邻居
        for neighbor in graph.get(state, []):
            if neighbor not in explored and neighbor not in frontier:
                parent[neighbor] = state
                frontier.append(neighbor)

    return None  # 没找到目标

# --- 测试 ---
if __name__ == '__main__':
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B', 'G'],
        'E': ['B', 'F', 'G'],
        'F': ['C', 'E', 'G'],
        'G': ['D', 'E', 'F']
    }

    start_state = 'A'
    goal_state = 'G'
    print(f"从节点 '{start_state}' 开始进行广度优先搜索:")
    path = bfs(graph, start_state, goal_state)
    print("最短路径:", path)
