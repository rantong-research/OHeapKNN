#!/usr/bin/env bash
#
# LAN.sh
#
# Configure the loopback interface for LAN experiments, optionally with an
# emulated RTT. The HTB classes shape the three Garnet ports (10000-10002).
#
# Usage:
#   ./LAN.sh          # LAN, no added delay
#   ./LAN.sh 5        # 5 ms RTT
#   ./LAN.sh 20       # 20 ms RTT

set -eu

RTT_MS="${1:-0}"

if ! [[ "$RTT_MS" =~ ^[0-9]+$ ]]; then
    echo "错误：RTT 必须为非负整数" >&2
    exit 1
fi

if ! command -v tc >/dev/null 2>&1; then
    echo "错误：找不到 tc 命令，请先安装 iproute2。" >&2
    exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
    echo "错误：找不到 sudo 命令。" >&2
    exit 1
fi

# Bidirectional communication: add RTT/2 delay in each direction.
DELAY_MS=$(awk -v rtt="$RTT_MS" 'BEGIN { printf "%.3f", rtt / 2 }')

# ===== Original network configuration =====

sudo tc qdisc del dev lo root 2>/dev/null || true

sudo tc qdisc add root dev lo handle 1: htb default 7

sudo tc class add dev lo parent 1:0 classid 1:1 htb rate 128Mbps
sudo tc class add dev lo parent 1:0 classid 1:2 htb rate 128Mbps
sudo tc class add dev lo parent 1:0 classid 1:3 htb rate 128Mbps
sudo tc class add dev lo parent 1:0 classid 1:4 htb rate 128Mbps
sudo tc class add dev lo parent 1:0 classid 1:5 htb rate 128Mbps
sudo tc class add dev lo parent 1:0 classid 1:6 htb rate 128Mbps
sudo tc class add dev lo parent 1:0 classid 1:7 htb rate 1000Mbps

sudo tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip sport 10000 0xffff flowid 1:1
sudo tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip dport 10000 0xffff flowid 1:2
sudo tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip sport 10001 0xffff flowid 1:3
sudo tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip dport 10001 0xffff flowid 1:4
sudo tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip sport 10002 0xffff flowid 1:5
sudo tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip dport 10002 0xffff flowid 1:6

# ===== RTT delay control =====

if [ "$RTT_MS" -gt 0 ]; then
    for i in {1..7}; do
        sudo tc qdisc add dev lo parent "1:$i" handle "$((10 + i)):" netem delay "${DELAY_MS}ms"
    done
fi

echo "网络配置完成！"
echo "目标 RTT: ${RTT_MS} ms"
echo "单向延迟: ${DELAY_MS} ms"

sudo tc -s qdisc show dev lo
