import random
import math
import matplotlib.pyplot as plt

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

def dist(node1, node2):
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

def steer(from_node, to_node, extend_length):
    new_node = Node(from_node.x, from_node.y)
    theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
    new_node.x += extend_length * math.cos(theta)
    new_node.y += extend_length * math.sin(theta)
    new_node.parent = from_node
    return new_node

def is_collision_free(node, obstacles):
    # 简化的碰撞检测：检查节点是否在任何圆形障碍物内
    for (ox, oy, r) in obstacles:
        if math.sqrt((node.x - ox)**2 + (node.y - oy)**2) <= r:
            return False
    return True

def rrt_planning(start, goal, obstacles, x_range, y_range, max_iter=500, extend_length=5):
    node_list = [start]

    for _ in range(max_iter):
        # 随机采样一个点
        rnd_node = Node(random.uniform(x_range[0], x_range[1]),
                        random.uniform(y_range[0], y_range[1]))

        # 找到树中最近的节点
        nearest_node = node_list[0]
        for node in node_list:
            if dist(node, rnd_node) < dist(nearest_node, rnd_node):
                nearest_node = node

        # 扩展树
        new_node = steer(nearest_node, rnd_node, extend_length)

        # 检查碰撞并添加到树中
        if is_collision_free(new_node, obstacles):
            node_list.append(new_node)

            # 检查是否到达目标
            if dist(new_node, goal) <= extend_length:
                if is_collision_free(goal, obstacles):
                    goal.parent = new_node
                    node_list.append(goal)
                    return node_list
    return None

def plot_path(node_list, obstacles, start, goal, x_range, y_range):
    plt.figure(figsize=(8, 8))
    for (ox, oy, r) in obstacles:
        circle = plt.Circle((ox, oy), r, color='gray', alpha=0.5)
        plt.gca().add_patch(circle)

    if node_list:
        path = []
        current = node_list[-1]
        while current.parent:
            path.append(current)
            current = current.parent
        path.append(current)
        path.reverse()

        for i in range(len(path) - 1):
            plt.plot([path[i].x, path[i+1].x], [path[i].y, path[i+1].y], '-g')

        for node in node_list:
            if node.parent:
                plt.plot([node.x, node.parent.x], [node.y, node.parent.y], '-b', alpha=0.5)

    plt.plot(start.x, start.y, 'or', markersize=10, label='Start')
    plt.plot(goal.x, goal.y, 'og', markersize=10, label='Goal')
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('RRT Motion Planning Example')
    plt.grid(True)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('course/01-robot-grasping-intro/images/rrt_path_planning.png')
    plt.show()

if __name__ == '__main__':
    start_node = Node(10, 10)
    goal_node = Node(90, 90)
    obstacles = [
        (30, 30, 10), (70, 70, 10), (30, 70, 10), (70, 30, 10)
    ]
    x_range = (0, 100)
    y_range = (0, 100)

    path_nodes = rrt_planning(start_node, goal_node, obstacles, x_range, y_range)

    if path_nodes:
        print("Path found!")
        plot_path(path_nodes, obstacles, start_node, goal_node, x_range, y_range)
    else:
        print("No path found.")
