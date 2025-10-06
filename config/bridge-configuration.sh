#bridge vlan interfaces

sudo ip link add br0 type bridge

sudo ip link set br0 up

sudo ip link set eth1 master br0

sudo ip link set eth0 master br0

sudo ip addr add 192.168.1.41/24 dev br0

bridge link show
