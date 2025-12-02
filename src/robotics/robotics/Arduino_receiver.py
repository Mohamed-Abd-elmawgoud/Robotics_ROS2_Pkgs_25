import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import serial

class SerialWriter(Node):
    def __init__(self):
        super().__init__('serial_writer')

        # Adjust this to your Arduino port
        self.serial_port = serial.Serial('/dev/ttyACM0', 115200, timeout=1)

        # Publish integers to Arduino every 0.5s
        self.timer = self.create_timer(0.5, self.send_data)
        self.counter = 0

    def send_data(self):
        self.counter += 1
        data = f"{self.counter}\n"           # add newline for Arduino readStringUntil
        self.serial_port.write(data.encode())  # send to Arduino
        self.get_logger().info(f"Sent to Arduino: {self.counter}")

def main(args=None):
    rclpy.init(args=args)
    node = SerialWriter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
