#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import csv


class SimpleSerialTransmitter(Node):
    def __init__(self):
        super().__init__("simple_serial_transmitter")

        self.declare_parameter("port", "/dev/ttyACM0")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("csv_file", "home_to_pick.csv")
        # self.declare_parameter("csv_file", "Pick_up.csv")
        self.declare_parameter("send_interval", 0.1)

        self.port_ = self.get_parameter("port").value
        self.baudrate_ = self.get_parameter("baudrate").value
        self.csv_file_ = self.get_parameter("csv_file").value
        self.send_interval_ = self.get_parameter("send_interval").value

        self.sub_ = self.create_subscription(String, "serial_transmitter", self.msgCallback, 10)
        self.arduino_ = serial.Serial(port=self.port_, baudrate=self.baudrate_, timeout=0.1)

        # Load angles from CSV
        self.angles_data_ = self.load_csv_data()
        self.current_row_ = 0
        self.sending_sequence_ = False
        self.sequence_count_ = 0

        # Timer for sending data
        self.send_timer_ = self.create_timer(self.send_interval_, self.sendAnglesCallback)
        
        # Timer for checking Arduino responses
        self.check_timer_ = self.create_timer(0.01, self.checkArduinoResponse)

        self.get_logger().info(f"Loaded {len(self.angles_data_)} rows from CSV")
        self.get_logger().info("Waiting for Arduino to request sequence with 'READY'")

    def load_csv_data(self):
        """Load angle data from CSV file"""
        angles_data = []
        try:
            with open(self.csv_file_, 'r') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    angles = [
                        float(row['q1']),
                        float(row['q2']),
                        float(row['q3']),
                        float(row['q4']),
                        float(row['q5'])
                    ]
                    angles_data.append(angles)
            self.get_logger().info(f"Successfully loaded {len(angles_data)} angle sets from {self.csv_file_}")
        except Exception as e:
            self.get_logger().error(f"Error loading CSV file: {e}")
        
        return angles_data

    def checkArduinoResponse(self):
        """Check for messages from Arduino"""
        if self.arduino_.in_waiting > 0:
            try:
                response = self.arduino_.readline().decode('utf-8').strip()
                if response:
                    self.get_logger().info(f"Arduino says: {response}")
                    
                    # Arduino sends "READY" when it wants a new sequence
                    if response == "READY":
                        self.get_logger().info("Arduino is ready, starting new sequence...")
                        self.current_row_ = 0
                        self.sending_sequence_ = True
                        self.sequence_count_ += 1
                        
            except Exception as e:
                self.get_logger().debug(f"Error reading Arduino response: {e}")

    def sendAnglesCallback(self):
        """Send angles to Arduino automatically"""
        if not self.sending_sequence_:
            return
        
        if not self.angles_data_:
            self.get_logger().warn("No angle data available")
            return
        
        if self.current_row_ >= len(self.angles_data_):
            # Sequence complete - send END flag
            self.sending_sequence_ = False
            end_message = "END\n"
            try:
                self.arduino_.write(end_message.encode('utf-8'))
                self.get_logger().info(f"=== Sequence {self.sequence_count_} Complete! Sent END flag ===")
                self.get_logger().info("Waiting for Arduino to send 'READY' for next sequence...")
            except Exception as e:
                self.get_logger().error(f"Error sending END flag: {e}")
            return
        
        angles = self.angles_data_[self.current_row_]
        
        # Format: "q1,q2,q3,q4,q5\n"
        message = f"{angles[0]:.2f},{angles[1]:.2f},{angles[2]:.2f},{angles[3]:.2f},{angles[4]:.2f}\n"
        
        try:
            self.arduino_.write(message.encode('utf-8'))
            self.get_logger().info(f"Seq {self.sequence_count_}, Step {self.current_row_}/{len(self.angles_data_)-1}: {message.strip()}")
        except Exception as e:
            self.get_logger().error(f"Error sending data: {e}")
        
        self.current_row_ += 1

    def msgCallback(self, msg):
        """Manual message callback"""
        self.get_logger().info("New message received, publishing on serial: %s" % self.arduino_.name)
        self.arduino_.write(msg.data.encode("utf-8"))


def main():
    rclpy.init()

    simple_serial_transmitter = SimpleSerialTransmitter()
    
    try:
        rclpy.spin(simple_serial_transmitter)
    except KeyboardInterrupt:
        pass
    finally:
        simple_serial_transmitter.arduino_.close()
        simple_serial_transmitter.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()