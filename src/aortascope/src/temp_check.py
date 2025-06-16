#!/usr/bin/env python3
import rospy
import subprocess
from std_msgs.msg import Float32
import time

def read_cpu_temp():
    """
    Return the Package id 0 temperature as a float (°C), or None on error.
    """
    
    result = subprocess.run(
        ['sensors'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=2.0
    )


    for line in result.stdout.splitlines():
        if 'Package id 0:' in line:
            temp = float(line.split('+')[1].split('°C')[0])
            print("CPU Temperature is:", temp)
            if(temp>96.0):
                rospy.set_param('/global_pause', 1)
                
                

               


    

def timer_callback(event):
    read_cpu_temp()
    
       

if __name__ == '__main__':
    rospy.init_node('temp_check', anonymous=False)

    # Fire timer every 5 seconds
    rospy.Timer(rospy.Duration(5.0), timer_callback)
    rospy.loginfo("cpu_temp_monitor node started, polling every 5 s")
    rospy.spin()