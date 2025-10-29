import heapq

def ucs(graph, start_node, goal_node):
    # 优先队列中的元素是 (代价, 当前节点, 路径)
    frontier = []
    # 用于记录每个节点在优先队列中的最小代价和路径
    node_info = {}
    visited = set()
    
    # 初始化起始节点
    heapq.heappush(frontier, (0, start_node, [start_node]))
    node_info[start_node] = (0, [start_node])
    
    while frontier:
        cost, current_node, path = heapq.heappop(frontier)

        if current_node in visited:
            continue
            
        visited.add(current_node)
        
        if current_node == goal_node:
            return path, cost

        for neighbor, weight in graph.get(current_node, []):
            if neighbor in visited:
                continue
                
            new_cost = cost + weight
            new_path = path + [neighbor]
            
            # 如果邻居节点不在node_info中，或者找到了更小的代价
            if neighbor not in node_info or new_cost < node_info[neighbor][0]:
                # 更新节点信息
                node_info[neighbor] = (new_cost, new_path)
                # 直接推入优先队列（可能有重复，但会被上面的visited检查过滤）
                heapq.heappush(frontier, (new_cost, neighbor, new_path))
    
    return None, None

# --- 测试案例 ---
if __name__ == '__main__':
    # 定义一个带权重的图
    graph = {
        'A': [('B', 1), ('C', 5)],
        'B': [('A', 1), ('D', 2), ('E', 4)],
        'C': [('A', 5), ('F', 6)],
        'D': [('B', 2)],
        'E': [('B', 4), ('F', 3)],
        'F': [('C', 6), ('E', 3)]
    }
    start_node = 'A'
    goal_node = 'F'

    print(f"从 '{start_node}' 到 '{goal_node}' 的一致代价搜索:")
    path, cost = ucs(graph, start_node, goal_node)
    
    if path:
        print("找到的最低代价路径:", path)
        print("总代价:", cost)
    else:
        print("未找到路径。")

    # 预期输出:
    # 从 'A' 到 'F' 的一致代价搜索:
    # 找到的最低代价路径: ['A', 'B', 'E', 'F']
    # 总代价: 8

    # --- 测试decreaseKey功能的案例 ---
    print("\n" + "="*50)
    print("测试decreaseKey功能的案例:")
    
    # 创建一个需要decreaseKey的图
    test_graph = {
        'A': [('B', 10), ('C', 1)],
        'B': [('D', 1)],
        'C': [('B', 1)],  # 这条路径 A->C->B 比 A->B 更优
        'D': [('E', 1)],
        'E': []
    }
    
    start = 'A'
    goal = 'E'
    
    path, cost = ucs(test_graph, start, goal)
    print(f"从 '{start}' 到 '{goal}':")
    print("路径:", path)
    print("代价:", cost)
    print("正确路径应该是: A->C->B->D->E (总代价=4)")