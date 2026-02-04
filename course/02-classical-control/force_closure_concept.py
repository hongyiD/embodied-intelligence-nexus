import matplotlib.pyplot as plt
import numpy as np

def plot_friction_cone(ax, contact_point, normal_vector, friction_coeff, scale=1.0):
    # 摩擦锥的半角
    friction_angle = np.arctan(friction_coeff)

    # 法线方向
    nx, ny = normal_vector / np.linalg.norm(normal_vector)

    # 摩擦锥的两个边界向量
    # 旋转法线向量 +/- 摩擦角
    R1 = np.array([[np.cos(friction_angle), -np.sin(friction_angle)],
                   [np.sin(friction_angle), np.cos(friction_angle)]])
    R2 = np.array([[np.cos(-friction_angle), -np.sin(-friction_angle)],
                   [np.sin(-friction_angle), np.cos(-friction_angle)]])

    v1 = R1 @ np.array([nx, ny])
    v2 = R2 @ np.array([nx, ny])

    # 绘制摩擦锥
    ax.plot([contact_point[0], contact_point[0] + v1[0] * scale],
            [contact_point[1], contact_point[1] + v1[1] * scale], 
            color='blue', linestyle='--', alpha=0.7)
    ax.plot([contact_point[0], contact_point[0] + v2[0] * scale],
            [contact_point[1], contact_point[1] + v2[1] * scale], 
            color='blue', linestyle='--', alpha=0.7)
    ax.plot([contact_point[0], contact_point[0] + nx * scale],
            [contact_point[1], contact_point[1] + ny * scale], 
            color='red', linewidth=2, alpha=0.8, label='Normal Force')

    # 绘制接触点
    ax.plot(contact_point[0], contact_point[1], 'ko', markersize=5)


if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Force Closure Concept: Friction Cones')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)

    # 模拟物体中心
    object_center = np.array([5, 5])
    ax.plot(object_center[0], object_center[1], 'x', color='purple', markersize=10, label='Object Center')

    # 接触点1
    contact1 = np.array([4, 5])
    normal1 = np.array([1, 0]) # 法线向右
    plot_friction_cone(ax, contact1, normal1, friction_coeff=0.5, scale=1.5)

    # 接触点2
    contact2 = np.array([6, 5])
    normal2 = np.array([-1, 0]) # 法线向左
    plot_friction_cone(ax, contact2, normal2, friction_coeff=0.5, scale=1.5)

    ax.legend()
    plt.savefig('course/02-classical-control/images/force_closure_concept.png')
    plt.show()
