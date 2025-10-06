sudo tc qdisc add dev eth0 clsact 2>/dev/null
sudo tc qdisc del dev eth1 parent root 2>/dev/null

# Example: VLAN 100, copy PCP 7/6/3/0 to same skb priority
sudo tc filter add dev eth0 ingress protocol 802.1Q flower vlan_id 100 vlan_prio 7 action skbedit priority 7
sudo tc filter add dev eth0 ingress protocol 802.1Q flower vlan_id 100 vlan_prio 6 action skbedit priority 6
sudo tc filter add dev eth0 ingress protocol 802.1Q flower vlan_id 100 vlan_prio 3 action skbedit priority 3
sudo tc filter add dev eth0 ingress protocol ip flower ip_proto udp dst_port 319 action skbedit priority 0
sudo tc filter add dev eth0 ingress protocol ip flower ip_proto udp dst_port 320 action skbedit priority 0

