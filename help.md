底盘 ros2 launch chassis_controller chassis_controller.launch.py
感知 ros2 launch track_perception_cpp fused_perception.launch.py
控制 ros2 launch line_follower_control_cpp controller.launch.py
键盘 ros2 run keyboard_controller keyboard_control_node


环境
source /opt/ros/humble/setup.bash
source ~/scuderiaferrari/install/setup.bash

四种速度从快到慢
./manettino/esc_off.sh
./manettino/tc_off.sh
./manettino/race.sh
./manettino/sport.sh
./manettino/wet.sh

发车
ros2 service call /line_follower/start std_srvs/srv/Trigger "{}"
停车
ros2 service call /line_follower/stop std_srvs/srv/Trigger "{}"
不可以ctrl + c


api导入方式（不能写在仓库里）：
read -rsp 'QIANFAN_API_KEY: ' QIANFAN_API_KEY
echo
export QIANFAN_API_KEY
ros2 launch track_perception_cpp fused_perception.launch.py