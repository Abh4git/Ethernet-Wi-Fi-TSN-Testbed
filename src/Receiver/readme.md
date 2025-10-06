# TSN Receiver  
Contains source code for listening to a WLAN interface and receiving packets directly from the interface.  
Generates traffic logs and pcap files containing capture.    

# Buid  
g++ receiver_wlan.cpp -o receiver_wlan -lpcap -lrt  
# Run 
sudo ./receiver_wlan

# Major Prerequisites  
## Mandatory  
sudo ip addr add 192.168.1.44/24 dev enp0s2  

sudo ptp4l -i enp0s2 -f /etc/linuxptp/ptp4l-client.conf -m  

## Optional  
sudo phc2sys -s /dev/ptp0 -c CLOCK_REALTIME -O 0 -m  


