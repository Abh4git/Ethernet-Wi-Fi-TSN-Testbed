#initial schedule - best working schedule
sudo tc qdisc replace dev eth1 parent root handle 100 taprio \
  num_tc 4 \
  map 3 2 2 2 1 1 1 0  3 2 2 2 1 1 1 0 \
  queues 1@0 1@1 1@2 1@3 \
  base-time 1000000000 \
  sched-entry S 0x0 13000 \
  sched-entry S 0x1 13000 \
  sched-entry S 0x0 24000 \
  sched-entry S 0x2 13000 \
  sched-entry S 0x4 12000 \
  sched-entry S 0x8 12000 \
  sched-entry S 0x0 124000 \
  flags 0x1 txtime-delay 0 clockid CLOCK_TAI
