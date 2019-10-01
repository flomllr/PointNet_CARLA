#! /bin/bash
args=("$@")
Model=${args[0]}
Modelpath="/usr/prakt/s0050/ss19_extendingpointnet/pointnet.pytorch/cls_final/${Model}.pth"
echo "Testing ${Model}"

# Settings
FOV=360
CAPTURE="--capture" # --capture
LIDAR="--lidar" # --lidar
VISUALIZE=true
# Town 1
# LeftPositions=(42 67 85 102 88 66 138)
# RightPositions=(42 67 85 102 88 66 138)
# StraightPositions=(66 138 78 14)

# LeftPositions=(42 88 66)
# RightPositions=(85 138)
# StraightPositions=(66 138)

# Town 2
# LeftPositions=(44 70 2 7 40)
# RightPositions=(44 70 2 7 40)
# StraightPositions=(7 40 61)
# LeftPositions=(49 2 7 40)
RightPositions=(78 44 1)
#FreeDrive=(70)


# Loop through left positions
for Pos in "${LeftPositions[@]}"
do
	echo Testing position $Pos, steering indicator left
	python pointnet_pilot.py --autopilot --use_steering_indicator --key_control --feature_transform --model="${Modelpath}" ${LIDAR} --position=${Pos} --frames=150 ${CAPTURE} --steering_indicator="left" --lidar_fov=${FOV}
	mv _capture/pos${Pos} _capture/pos${Pos}_left
done

# Loop through right positions
for Pos in "${RightPositions[@]}"
do
	echo Testing position $Pos, steering indicator right
	python pointnet_pilot.py --autopilot --use_steering_indicator --key_control --feature_transform --model="${Modelpath}" ${LIDAR} --position=${Pos} --frames=150 ${CAPTURE} --steering_indicator="right" --lidar_fov=${FOV}
	mv _capture/pos${Pos} _capture/pos${Pos}_right
done

# Loop through straight positions
for Pos in "${StraightPositions[@]}"
do
	echo Testing position $Pos, steering indicator straight
	python pointnet_pilot.py --autopilot --use_steering_indicator --key_control --feature_transform --model="${Modelpath}" ${LIDAR} --position=${Pos} --frames=150 ${CAPTURE} --steering_indicator="straight" --lidar_fov=${FOV}
	mv _capture/pos${Pos} _capture/pos${Pos}_straight
done

# No steering indicator set
for Pos in "${FreeDrive[@]}"
do
	echo Testing position $Pos, steering indicator straight
  echo ${FOV}
	python pointnet_pilot.py --autopilot --use_steering_indicator --key_control --feature_transform --model="${Modelpath}" ${LIDAR} --position=${Pos} --frames=150 ${CAPTURE} --lidar_fov=${FOV}
	mv _capture/pos${Pos} _capture/pos${Pos}_free
done


# Move visualisations to new folder
if [$CAPTURE == "--capture"];
then
  mv _capture _capture_${Model}
fi

case $VISUALIZE in
  (true)
    echo "Visualizing"
    # Start visualizing runs
    python visualize_run.py --lidar --gif --root=_capture_${Model}

    # delete everything except the gifs
    find _capture_${Model} -type f ! -name '*.gif' -exec rm -rf {} \;
  (false)
    echo "Not visualizing"
esac
echo Done.

