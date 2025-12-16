#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import csv


class MultiTrajectoryTransmitter(Node):
    def __init__(self):
        super().__init__("multi_trajectory_transmitter")

        self.declare_parameter("port", "/dev/ttyACM0")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("trajectory_files", ["Pickup_pos.csv", "Pick.csv", "Pick_reversed.csv", "Place_pos.csv", "Place.csv"])
        self.declare_parameter("send_interval", 0.05)

        self.port_ = self.get_parameter("port").value
        self.baudrate_ = self.get_parameter("baudrate").value
        self.trajectory_files_ = self.get_parameter("trajectory_files").value
        self.send_interval_ = self.get_parameter("send_interval").value

        self.sub_ = self.create_subscription(String, "serial_transmitter", self.msgCallback, 10)
        self.arduino_ = serial.Serial(port=self.port_, baudrate=self.baudrate_, timeout=0.1)

        # Load all trajectory files
        self.all_trajectories_ = []
        for traj_file in self.trajectory_files_:
            traj_data = self.load_csv_data(traj_file)
            if traj_data:
                self.all_trajectories_.append({
                    'filename': traj_file,
                    'data': traj_data
                })
        
        self.num_trajectories_ = len(self.all_trajectories_)
        self.current_trajectory_ = 0
        self.current_row_ = 0
        self.sending_sequence_ = False
        self.init_sent_ = False
        self.cycle_count_ = 0

        # Timer for sending data
        self.send_timer_ = self.create_timer(self.send_interval_, self.sendAnglesCallback)
        
        # Timer for checking Arduino responses
        self.check_timer_ = self.create_timer(0.01, self.checkArduinoResponse)

        self.get_logger().info(f"Loaded {self.num_trajectories_} trajectory files:")
        for i, traj in enumerate(self.all_trajectories_):
            self.get_logger().info(f"  Trajectory {i+1}: {traj['filename']} ({len(traj['data'])} steps)")
        self.get_logger().info("Waiting for Arduino READY signal...")

    def load_csv_data(self, csv_file):
        """Load angle data from CSV file"""
        angles_data = []
        try:
            with open(csv_file, 'r') as file:
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
            self.get_logger().info(f"Successfully loaded {len(angles_data)} angle sets from {csv_file}")
        except Exception as e:
            self.get_logger().error(f"Error loading CSV file {csv_file}: {e}")
        
        return angles_data

    def checkArduinoResponse(self):
        """Check for messages from Arduino"""
        if self.arduino_.in_waiting > 0:
            try:
                response = self.arduino_.readline().decode('utf-8').strip()
                if response:
                    self.get_logger().info(f"Arduino says: {response}")
                    
                    # First READY - send initialization with number of trajectories
                    if response == "READY" and not self.init_sent_:
                        init_message = f"INIT {self.num_trajectories_}\n"
                        self.arduino_.write(init_message.encode('utf-8'))
                        self.get_logger().info(f"Sent INIT: {self.num_trajectories_} trajectories")
                        self.init_sent_ = True
                        
                    # Subsequent READY - start next trajectory
                    elif response == "READY" and self.init_sent_:
                        if self.current_trajectory_ < self.num_trajectories_:
                            traj = self.all_trajectories_[self.current_trajectory_]
                            self.get_logger().info(
                                f"Arduino ready, starting Trajectory {self.current_trajectory_ + 1}/{self.num_trajectories_}: {traj['filename']}"
                            )
                            self.current_row_ = 0
                            self.sending_sequence_ = True
                        else:
                            self.get_logger().info("All trajectories sent in this cycle!")
                        
            except Exception as e:
                self.get_logger().debug(f"Error reading Arduino response: {e}")

    def sendAnglesCallback(self):
        """Send angles to Arduino automatically"""
        if not self.sending_sequence_:
            return
        
        if self.current_trajectory_ >= self.num_trajectories_:
            return
        
        traj = self.all_trajectories_[self.current_trajectory_]
        angles_data = traj['data']
        
        if not angles_data:
            self.get_logger().warn(f"No angle data available for trajectory {self.current_trajectory_ + 1}")
            return
        
        if self.current_row_ >= len(angles_data):
            # Trajectory complete - send END flag
            self.sending_sequence_ = False
            end_message = "END\n"
            try:
                self.arduino_.write(end_message.encode('utf-8'))
                self.get_logger().info(
                    f"=== Trajectory {self.current_trajectory_ + 1}/{self.num_trajectories_} Complete! Sent END flag ==="
                )
                
                self.current_trajectory_ += 1
                
                # Check if all trajectories are done
                if self.current_trajectory_ >= self.num_trajectories_:
                    self.cycle_count_ += 1
                    self.get_logger().info(f"=== Cycle {self.cycle_count_} Complete! All {self.num_trajectories_} trajectories executed ===")
                    self.get_logger().info("Waiting for Arduino READY to start new cycle...")
                    self.current_trajectory_ = 0  # Reset for next cycle
                    self.init_sent_ = False  # Need to send INIT again
                else:
                    self.get_logger().info(f"Waiting for Arduino READY for Trajectory {self.current_trajectory_ + 1}...")
                    
            except Exception as e:
                self.get_logger().error(f"Error sending END flag: {e}")
            return
        
        angles = angles_data[self.current_row_]
        
        # Format: "q1,q2,q3,q4,q5\n"
        message = f"{angles[0]:.2f},{angles[1]:.2f},{angles[2]:.2f},{angles[3]:.2f},{angles[4]:.2f}\n"
        
        try:
            self.arduino_.write(message.encode('utf-8'))
            self.get_logger().info(
                f"Traj {self.current_trajectory_ + 1}/{self.num_trajectories_}, "
                f"Step {self.current_row_}/{len(angles_data)-1}: {message.strip()}"
            )
        except Exception as e:
            self.get_logger().error(f"Error sending data: {e}")
        
        self.current_row_ += 1

    def msgCallback(self, msg):
        """Manual message callback"""
        self.get_logger().info("New message received, publishing on serial: %s" % self.arduino_.name)
        self.arduino_.write(msg.data.encode("utf-8"))


def main():
    rclpy.init()

    multi_trajectory_transmitter = MultiTrajectoryTransmitter()
    
    try:
        rclpy.spin(multi_trajectory_transmitter)
    except KeyboardInterrupt:
        pass
    finally:
        multi_trajectory_transmitter.arduino_.close()
        multi_trajectory_transmitter.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()