to run the gazebo world with buggy 
ros2 launch b3rb_gz_bringup sil.launch.py world:=Raceway_1

then to spawn the sign board in front of buggy

ros2 run ros_gz_sim create -world default -file ~/cognipilot/cranium/install/dream_world/share/dream_world/models/sign_board_1/model.sdf -name sign_test -x 1 -y -2 -z 0.5 -Y 1.57

then to run detector.py

cd ~/Downloads

python3 detector.py --goal A --image SignBoard.png



ros2 topic pub --once /mission/goal std_msgs/msg/String "{data: 'A'}"



ros2 topic pub -r 20 /cerebri/in/joy sensor_msgs/msg/Joy "{axes: [0.0, 0.5, 0.0, 0.0], buttons: [1, 0, 0, 0, 0, 0, 0, 1]}"


ros2 run b3rb_ros_line_follower detect


cd ~/cognipilot/cranium/src/b3rb_ros_line_follower/b3rb_ros_line_follower

python3 keyboard_teleop.py

