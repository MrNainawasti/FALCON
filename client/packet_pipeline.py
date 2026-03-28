from __future__ import annotations
import csv
import json
import socket
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
import dpkt

# Data Types

@dataclass(frozen=True)
class FlowKey:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int


@dataclass
class PacketInfo:
    timestamp: float
    length: int
    is_fwd: bool
    syn: bool
    ack: bool
    fin: bool
    rst: bool


@dataclass
class FlowFeatures:
    start_time: float = 0.0
    end_time: float = 0.0
    fwd_packets: int = 0
    bwd_packets: int = 0
    fwd_bytes: int = 0
    bwd_bytes: int = 0
    min_packet_size: int = 2**64 - 1
    max_packet_size: int = 0
    total_packet_size: int = 0
    sum_sq_packet_size: float = 0.0
    total_fwd_packet_size: int = 0
    total_bwd_packet_size: int = 0
    max_fwd_packet_size: int = 0
    min_bwd_packet_size: int = 2**64 - 1
    syn_count: int = 0
    ack_count: int = 0
    fin_count: int = 0
    rst_count: int = 0
    fwd_iat_total: float = 0.0
    bwd_iat_total: float = 0.0
    flow_iat_total: float = 0.0
    flow_iat_sum_sq: float = 0.0
    active_time_total: float = 0.0
    active_count: int = 0
    idle_time_total: float = 0.0
    idle_count: int = 0
    last_fwd_time: float = 0.0
    last_bwd_time: float = 0.0
    last_flow_time: float = 0.0
    current_active_start: float = 0.0


@dataclass
class JsonFlowOutput:
    flow_id: str
    features: list[float]


def update_features(features: FlowFeatures, packet: PacketInfo) -> None:
    """Update flow features from packet."""
    if features.fwd_packets == 0 and features.bwd_packets == 0:
        features.start_time = packet.timestamp
        features.last_flow_time = packet.timestamp
        features.current_active_start = packet.timestamp
        features.min_packet_size = packet.length
        features.max_packet_size = packet.length

    features.end_time = packet.timestamp
    features.min_packet_size = min(features.min_packet_size, packet.length)
    features.max_packet_size = max(features.max_packet_size, packet.length)
    features.total_packet_size += packet.length
    features.sum_sq_packet_size += float(packet.length * packet.length)

    if packet.syn:
        features.syn_count += 1
    if packet.ack:
        features.ack_count += 1
    if packet.fin:
        features.fin_count += 1
    if packet.rst:
        features.rst_count += 1

    flow_iat = packet.timestamp - features.last_flow_time

    if (features.fwd_packets + features.bwd_packets) > 0 and flow_iat > 5.0:
        features.idle_time_total += flow_iat
        features.idle_count += 1
        active_time = features.last_flow_time - features.current_active_start
        features.active_time_total += active_time
        features.active_count += 1
        features.current_active_start = packet.timestamp

    if (features.fwd_packets + features.bwd_packets) > 0:
        features.flow_iat_total += flow_iat
        features.flow_iat_sum_sq += flow_iat * flow_iat

    features.last_flow_time = packet.timestamp

    if packet.is_fwd:
        features.max_fwd_packet_size = max(features.max_fwd_packet_size, packet.length)
        features.total_fwd_packet_size += packet.length
        features.fwd_packets += 1
        features.fwd_bytes += packet.length
        if features.fwd_packets > 1:
            iat = packet.timestamp - features.last_fwd_time
            features.fwd_iat_total += iat
        features.last_fwd_time = packet.timestamp
    else:
        features.min_bwd_packet_size = min(features.min_bwd_packet_size, packet.length)
        features.total_bwd_packet_size += packet.length
        features.bwd_packets += 1
        features.bwd_bytes += packet.length
        if features.bwd_packets > 1:
            iat = packet.timestamp - features.last_bwd_time
            features.bwd_iat_total += iat
        features.last_bwd_time = packet.timestamp


def select_features(features: FlowFeatures, configured_features: list[str]) -> list[float]:
    """Extract selected features from computed flow."""
    out: list[float] = []
    for raw_name in configured_features:
        name = raw_name.strip()
        if name == "Flow Duration":
            val = features.end_time - features.start_time
        elif name in {"Total Fwd Packets", "fwd_packets"}:
            val = float(features.fwd_packets)
        elif name in {"Total Backward Packets", "bwd_packets"}:
            val = float(features.bwd_packets)
        elif name in {"Total Length of Fwd Packets", "fwd_bytes"}:
            val = float(features.fwd_bytes)
        elif name in {"Total Length of Bwd Packets", "bwd_bytes"}:
            val = float(features.bwd_bytes)
        elif name in {"Flow Bytes/s", "bytes_per_second"}:
            duration = features.end_time - features.start_time
            total_bytes = features.fwd_bytes + features.bwd_bytes
            val = (float(total_bytes) / duration) if duration > 0.0 else 0.0
        elif name in {"Flow Packets/s", "packets_per_second"}:
            duration = features.end_time - features.start_time
            total_pkts = features.fwd_packets + features.bwd_packets
            val = (float(total_pkts) / duration) if duration > 0.0 else 0.0
        elif name in {"FIN Flag Count", "fin_count"}:
            val = float(features.fin_count)
        elif name in {"SYN Flag Count", "syn_count"}:
            val = float(features.syn_count)
        elif name in {"ACK Flag Count", "ack_count"}:
            val = float(features.ack_count)
        elif name in {"Packet Length Mean", "mean_packet_size"}:
            total_pkts = features.fwd_packets + features.bwd_packets
            val = (float(features.total_packet_size) / float(total_pkts)) if total_pkts > 0 else 0.0
        elif name == "Fwd Packet Length Mean":
            val = (
                float(features.total_fwd_packet_size) / float(features.fwd_packets)
                if features.fwd_packets > 0
                else 0.0
            )
        elif name == "Bwd Packet Length Mean":
            val = (
                float(features.total_bwd_packet_size) / float(features.bwd_packets)
                if features.bwd_packets > 0
                else 0.0
            )
        elif name == "Fwd Packet Length Max":
            val = float(features.max_fwd_packet_size)
        elif name == "Bwd Packet Length Min":
            val = 0.0 if features.min_bwd_packet_size == (2**64 - 1) else float(features.min_bwd_packet_size)
        elif name == "Down/Up Ratio":
            val = float(features.bwd_packets) / float(features.fwd_packets) if features.fwd_packets > 0 else 0.0
        elif name == "Flow IAT Mean":
            total_pkts = features.fwd_packets + features.bwd_packets
            val = features.flow_iat_total / (float(total_pkts) - 1.0) if total_pkts > 1 else 0.0
        elif name == "Flow IAT Std":
            total_pkts = features.fwd_packets + features.bwd_packets
            if total_pkts > 1:
                n = float(total_pkts - 1)
                mean = features.flow_iat_total / n
                variance = (features.flow_iat_sum_sq / n) - (mean * mean)
                val = variance**0.5 if variance > 0.0 else 0.0
            else:
                val = 0.0
        elif name == "Active Mean":
            final_burst = features.last_flow_time - features.current_active_start
            active_total = features.active_time_total + final_burst
            count = features.active_count
            if final_burst > 0.0:
                count += 1
            val = active_total / float(count) if count > 0 else 0.0
        elif name == "Idle Mean":
            val = features.idle_time_total / float(features.idle_count) if features.idle_count > 0 else 0.0
        elif name == "Subflow Fwd Bytes":
            val = float(features.fwd_bytes)
        else:
            val = 0.0
        out.append(val)
    return out


def parse_packet(packet_data: bytes, timestamp: float) -> tuple[FlowKey, PacketInfo] | None:
    """Parse Ethernet packet and extract flow key + packet info."""
    try:
        eth = dpkt.ethernet.Ethernet(packet_data)
    except (dpkt.NeedData, dpkt.UnpackError):
        return None

    ip = eth.data
    src_ip: str
    dst_ip: str
    protocol: int

    if isinstance(ip, dpkt.ip.IP):
        src_ip = socket.inet_ntoa(ip.src)
        dst_ip = socket.inet_ntoa(ip.dst)
        protocol = int(ip.p)
    elif isinstance(ip, dpkt.ip6.IP6):
        src_ip = socket.inet_ntop(socket.AF_INET6, ip.src)
        dst_ip = socket.inet_ntop(socket.AF_INET6, ip.dst)
        protocol = int(ip.nxt)
    else:
        return None

    transport = ip.data
    syn = False
    ack = False
    fin = False
    rst = False

    if isinstance(transport, dpkt.tcp.TCP):
        src_port = int(transport.sport)
        dst_port = int(transport.dport)
        flags = int(transport.flags)
        syn = bool(flags & dpkt.tcp.TH_SYN)
        ack = bool(flags & dpkt.tcp.TH_ACK)
        fin = bool(flags & dpkt.tcp.TH_FIN)
        rst = bool(flags & dpkt.tcp.TH_RST)
    elif isinstance(transport, dpkt.udp.UDP):
        src_port = int(transport.sport)
        dst_port = int(transport.dport)
    else:
        return None

    key = FlowKey(src_ip=src_ip, dst_ip=dst_ip, src_port=src_port, dst_port=dst_port, protocol=protocol)
    info = PacketInfo(
        timestamp=timestamp, length=len(packet_data), is_fwd=True, syn=syn, ack=ack, fin=fin, rst=rst
    )
    return key, info


# Pipeline

def process_csv(csv_path: str, config: dict, output_path: str) -> None:
    """Extract features from CSV."""
    exported: list[JsonFlowOutput] = []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader, [])
        header_map = {h.strip().lower(): i for i, h in enumerate(headers)}

        print("CSV loaded. Extracting configured features...")
        for row_idx, row in enumerate(reader):
            features_vals = []
            for feature_name in config["features"]:
                value = 0.0
                for alias in [feature_name.strip().lower()]:
                    idx = header_map.get(alias)
                    if idx is not None and idx < len(row):
                        try:
                            value = float(row[idx].strip())
                            break
                        except ValueError:
                            value = 0.0
                features_vals.append(value)

            flow_id = ""
            for key in ("flow_id", "flow id"):
                idx = header_map.get(key)
                if idx is not None and idx < len(row) and row[idx].strip():
                    flow_id = row[idx].strip()
                    break
            if not flow_id:
                flow_id = f"csv_row_{row_idx + 1}"

            exported.append(JsonFlowOutput(flow_id=flow_id, features=features_vals))

    Path(output_path).write_text(json.dumps([asdict(r) for r in exported], indent=2), encoding="utf-8")
    print(f"Exported {len(exported)} rows with {len(config['features'])} features each to {output_path}")


def process_pcap(pcap_path: str, config: dict, output_path: str) -> None:
    """Extract flows and features from PCAP."""
    flows: dict[FlowKey, FlowFeatures] = {}

    f = open(pcap_path, "rb")
    try:
        reader = dpkt.pcap.Reader(f)
    except (ValueError, dpkt.dpkt.Error):
        f.seek(0)
        reader = dpkt.pcapng.Reader(f)

    print("Parsing packets and building flows...")
    for ts, data in reader:
        parsed = parse_packet(data, float(ts))
        if parsed is None:
            continue

        key, packet = parsed
        is_fwd = True
        actual_key = key

        rev_key = FlowKey(
            src_ip=key.dst_ip,
            dst_ip=key.src_ip,
            src_port=key.dst_port,
            dst_port=key.src_port,
            protocol=key.protocol,
        )

        if rev_key in flows and key not in flows:
            is_fwd = False
            actual_key = rev_key

        entry = flows.setdefault(actual_key, FlowFeatures())
        packet_info = PacketInfo(
            timestamp=packet.timestamp,
            length=packet.length,
            is_fwd=is_fwd,
            syn=packet.syn,
            ack=packet.ack,
            fin=packet.fin,
            rst=packet.rst,
        )
        update_features(entry, packet_info)

    f.close()

    print("Flows built. Extracting configured features...")
    exported: list[JsonFlowOutput] = []
    for key, flow_features in flows.items():
        selected = select_features(flow_features, config["features"])
        flow_id = f"{key.src_ip}-{key.dst_ip}-{key.src_port}-{key.dst_port}-{key.protocol}"
        exported.append(JsonFlowOutput(flow_id=flow_id, features=selected))

    Path(output_path).write_text(json.dumps([asdict(r) for r in exported], indent=2), encoding="utf-8")
    print(f"Exported {len(exported)} flows with {len(config['features'])} features each to {output_path}")


def main():
    if len(sys.argv) != 4:
        print("Usage: python main.py <input.(pcap|csv)> <config.json> <output.json>", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    config_path = sys.argv[2]
    output_path = sys.argv[3]

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    is_csv = Path(input_path).suffix.lower() == ".csv"

    if is_csv:
        process_csv(input_path, config, output_path)
    else:
        process_pcap(input_path, config, output_path)


if __name__ == "__main__":
    main()