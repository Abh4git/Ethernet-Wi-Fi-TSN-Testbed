# TSN Sender source code used in Linux Ubuntu 22.04 and 24.04 LTS environments
1. sender_wlan.cpp
2. .json file used as configuration
3. _input_data folder containing data files used for sending


# Building Sender Application  
g++ sender_wlan.cpp -o sender_wlan -lpthread -lrt  

# Prerequisites  
Running PTP Synchronization , Correct IP Set  

## setupenvironment  

sudo ip add add 192.168.1.50/24 dev enp3s0    
### Mandatory - Disable firwalls and Start PTP service    
sudo systemctl stop firewalld  
sudo ufw disable  
ifconfig   
sudo ptp4l -i enp3s0 -f /etc/linuxptp/ptp4l-master.conf -m   
### Optional  
sudo phc2sys -s /dev/ptp0 -c CLOCK_REALTIME -O 0 -m  

