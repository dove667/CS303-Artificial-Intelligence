def dls(graph, start_state, goal_state, limit):
    def recurse(state, depth, path_set):
        if depth > limit:
            return None
        if state == goal_state:
            return [state]
        for neighbor in graph.get(state, []):
            if neighbor in path_set:  # 避免回路
                continue
            path_set.add(neighbor)
            res = recurse(neighbor, depth + 1, path_set)
            if res:
                return [state] + res
            path_set.discard(neighbor)
        return None

    return recurse(start_state, 0, set([start_state]))

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
    limit = 3
    print(f"从节点 '{start_state}' 开始进行深度限制搜索 (limit={limit}):")
    path = dls(graph, start_state, goal_state, limit)
    print("路径:", path)
# ...existing code...