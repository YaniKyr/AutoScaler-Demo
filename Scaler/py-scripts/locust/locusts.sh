#!/bin/bash

source ~/locust/bin/activate
while true
do
	echo "Startin locust at $(date)."
	locust -f locust.py --headless -t 2h
done
