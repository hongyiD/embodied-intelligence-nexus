import matplotlib.pyplot as plt

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0

    def update(self, current_value, dt):
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

# 模拟机器人关节
class RobotJoint:
    def __init__(self, initial_position=0.0):
        self.position = initial_position
        self.velocity = 0.0
        self.acceleration = 0.0

    def step(self, control_input, dt):
        # 简单的动力学模型：控制输入直接影响加速度
        self.acceleration = control_input * 0.1 # 假设一个简单的比例关系
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        return self.position

if __name__ == "__main__":
    # PID参数
    Kp = 0.5
    Ki = 0.1
    Kd = 0.05
    setpoint = 10.0 # 目标位置

    pid = PIDController(Kp, Ki, Kd, setpoint)
    joint = RobotJoint(initial_position=0.0)

    time_steps = 200
    dt = 0.1 # 时间步长

    positions = []
    times = []

    for i in range(time_steps):
        control_output = pid.update(joint.position, dt)
        current_position = joint.step(control_output, dt)

        positions.append(current_position)
        times.append(i * dt)

    plt.figure(figsize=(10, 6))
    plt.plot(times, positions, label='Joint Position')
    plt.axhline(y=setpoint, color='r', linestyle='--', label='Setpoint')
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.title('PID Control of Robot Joint Position')
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('course/02-classical-control/images/pid_control_example.png')
    plt.show()
