#!/usr/bin/env bash
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 pressure $1 temperature" >&2
  exit 1
fi
cmd='curl -X POST 'http://192.168.0.11/manage_inventory/rpi0/' -d 'reporting_location=raspberry' -d 'pressure="$1"' -d 'temperature="$2"''
$cmd