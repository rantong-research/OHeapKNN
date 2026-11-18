#!/usr/bin/env bash
#
# WAN.sh
#
# Configure the loopback interface for WAN experiments: 5 Mbps HTB classes
# for the three Garnet ports (10000-10002) and 20 ms one-way delay, i.e.
# approximately 40 ms RTT.

set -eu

if ! command -v tc >/dev/null 2>&1; then
    echo "错误：找不到 tc 命令，请先安装 iproute2。" >&2
    exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
    echo "错误：找不到 sudo 命令。" >&2
    exit 1
fi

sudo tc qdisc del dev lo root 2>/dev/null || true
sudo tc qdisc add root dev lo handle 1: htb default 7

sudo tc class add dev lo parent 1:0 classid 1:1 htb rate 5Mbps
sudo tc class add dev lo parent 1:0 classid 1:2 htb rate 5Mbps
sudo tc class add dev lo parent 1:0 classid 1:3 htb rate 5Mbps
sudo tc class add dev lo parent 1:0 classid 1:4 htb rate 5Mbps
sudo tc class add dev lo parent 1:0 classid 1:5 htb rate 5Mbps
sudo tc class add dev lo parent 1:0 classid 1:6 htb rate 5Mbps
sudo tc class add dev lo parent 1:0 classid 1:7 htb rate 1000Mbps

sudo tc qdisc add dev lo parent 1:1 handle 2:0 netem delay 20ms
sudo tc qdisc add dev lo parent 1:2 handle 3:0 netem delay 20ms
sudo tc qdisc add dev lo parent 1:3 handle 4:0 netem delay 20ms
sudo tc qdisc add dev lo parent 1:4 handle 5:0 netem delay 20ms
sudo tc qdisc add dev lo parent 1:5 handle 6:0 netem delay 20ms
sudo tc qdisc add dev lo parent 1:6 handle 7:0 netem delay 20ms

sudo tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip sport 10000 0xffff flowid 1:1
sudo tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip dport 10000 0xffff flowid 1:2
sudo tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip sport 10001 0xffff flowid 1:3
sudo tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip dport 10001 0xffff flowid 1:4
sudo tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip sport 10002 0xffff flowid 1:5
sudo tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip dport 10002 0xffff flowid 1:6

echo "网络配置完成！"
echo "目标 RTT: 40 ms"
echo "单向延迟: 20 ms"

sudo tc -s qdisc show dev lo
