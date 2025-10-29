from DLS import dls 

def ids(graph, start_state, goal_state, max_depth):
    for i in range(max_depth + 1):
        print(f"尝试深度: {i}")
        result = dls(graph, start_state, goal_state, i)
        if result is not None:
            return result
    return None


if __name__ == '__main__':
    graph = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F', 'G'],
        'D': [],
        'E': [],
        'F': [],
        'G': ['H'],
        'H': []
    }

    start_state = 'A'
    goal_state = 'H'
    max_depth = 4

    print(f"从 '{start_state}' 到 '{goal_state}' 的迭代加深搜索:")
    path = ids(graph, start_state, goal_state, max_depth)
    
    if path:
        print("\n找到的路径:", path)
    else:
        print(f"\n在最大深度 {max_depth} 内未找到路径。")

    # 预期输出:
    # 从 'A' 到 'H' 的迭代加深搜索:
    # 尝试深度: 0
    # 尝试深度: 1
    # 尝试深度: 2
    # 尝试深度: 3
    #
    # 找到的路径: ['A', 'C', 'G', 'H']