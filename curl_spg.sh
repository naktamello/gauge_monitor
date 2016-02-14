#!/usr/bin/env bash
#if [ "$#" -ne 2 ]; then
#  echo "Usage: $0 pressure $1 temperature" >&2
#  exit 1
#fi
#cmd='curl -X POST 'http://192.168.0.11/manage_inventory/rpi0/' -d 'reporting_location=raspberry' -d 'pressure="$1"' -d 'temperature="$2"''
pressure=$(sed -n -e 's/^.*pressure://p' gauge_data.txt)
temperature=$(sed -n -e 's/^.*temperature://p' gauge_data.txt)
#echo "pressure = $pressure"
#echo "temperature = $temperature"
cmd='curl -X POST 'http://192.168.0.11/manage_inventory/rpi0/' -d 'reporting_location=raspberry' -d 'pressure="$pressure"' -d 'temperature="$temperature"''
$cmd