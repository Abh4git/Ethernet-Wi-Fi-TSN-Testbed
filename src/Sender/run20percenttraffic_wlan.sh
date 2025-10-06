#! /bin/bash

./raw_vlan_udp_sender_fixed_wlan > sender_output.log 2>&1 &
echo " Started vlan qos sender v3"
#tcpreplay --intf1=enp3s0 --mbps=200 --loop=8 input_data/onlyefhr-besteffort-traffic.pcap > tcpreplay_output.log 2>&1 &

#Number of parallel processes  to launch -9 for 100Mbps
num_processes=2 #2 #9

#Command to run your program 

program="./extraehrtraffic_wlan" 

for i in $(seq 1 $num_processes); do
    echo " Starting instance $i" 
    $program 2>&1 &
done
 
echo "Launched $num_processes parallel instances."

#allow manual termination



