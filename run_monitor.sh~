#!/bin/bash
control_c()
# run if user hits control-c
{
  echo -en "\n*** Ouch! Exiting ***\n"
  exit $?
}
 
ps cax | grep fbcp > /dev/null
if [ $? -eq 0 ]; then
  echo "Process is running."
else
  echo "Process is not running."
  /home/pi/rpi-fbcp/build/fbcp &
fi
cd /home/pi/gauge_monitor
python gauge_det.py
if [ -e "canvas.jpg" ]
then
    ps cax | grep fbi > /dev/null
    if pgrep fbi >/dev/null
    then
        echo "kill running fbi"
        sudo killall fbi
    fi
    sudo fbi -T 1 canvas.jpg &
fi
trap control_c SIGINT
while [ 1 ]
do
    read -n 1 -p -s key
        python gauge_det.py
        echo "running"
        if [ -e "canvas.jpg" ]
            then
            ps cax | grep fbi > /dev/null
            if pgrep fbi >/dev/null
            then
                echo "kill running fbi"
                sudo killall fbi
            fi
        sudo fbi -T 1 canvas.jpg
        fi
done
