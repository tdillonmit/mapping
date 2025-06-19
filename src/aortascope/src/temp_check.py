#!/usr/bin/env python3
import rospy
import subprocess
from std_msgs.msg import Float32
import time
from std_msgs.msg import Bool

global_pause_pub            = rospy.Publisher('/global_pause', Bool, queue_size=1)

                
def read_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = float(f.read()) / 1000.0  # millidegrees to degrees
            print("CPU Temperature (non blocking) is:", temp)
            if temp > 97.0:
                global_pause_pub.publish(True)
                time.sleep(5)
                global_pause_pub.publish(False)
    except Exception as e:
        rospy.logwarn(f"Could not read CPU temp: {e}")
               

def timer_callback(event):
    read_cpu_temp()
    
       

if __name__ == '__main__':
    rospy.init_node('temp_check', anonymous=False)

    # Fire timer every 5 seconds
    rospy.Timer(rospy.Duration(5.0), timer_callback)
    rospy.loginfo("cpu_temp_monitor node started, polling every 5 s")
    rospy.spin()