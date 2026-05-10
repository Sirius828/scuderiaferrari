"""
UWB数据帧解析器
解析LinkTrack NodeFrame2格式的UWB定位数据
"""

import struct
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class AnchorObservation:
    """锚点观测数据"""
    role: int  # 角色
    id: int  # ID
    distance_m: float  # 距离（米）
    fp_rssi_db: float  # 首径RSSI (dB)
    rx_rssi_db: float  # 接收RSSI (dB)


@dataclass
class NodeFrame2:
    """NodeFrame2数据帧"""
    role: int  # 角色
    id: int  # 标签ID
    system_time_ms: int  # 系统时间（毫秒）
    eop_x: float  # X轴误差范围
    eop_y: float  # Y轴误差范围
    eop_z: float  # Z轴误差范围
    pos_x: float  # X坐标（米）
    pos_y: float  # Y坐标（米）
    pos_z: float  # Z坐标（米）
    vel_x: float  # X速度（m/s）
    vel_y: float  # Y速度（m/s）
    vel_z: float  # Z速度（m/s）
    local_time_ms: int  # 本地时间（毫秒）
    voltage_v: float  # 电压（伏特）
    valid_node_quantity: int  # 有效锚点数量
    anchors: List[AnchorObservation]  # 锚点观测列表


class UWBFrameParser:
    """UWB数据帧解析器"""
    
    FRAME_HEADER = bytes([0x55, 0x04])  # 帧头
    MIN_FRAME_LENGTH = 120  # 最小帧长度
    MAX_FRAME_LENGTH = 512  # 最大帧长度
    
    def __init__(self):
        self.rx_buffer = bytearray()
    
    def add_data(self, data: bytes):
        """添加接收到的数据到缓冲区"""
        self.rx_buffer.extend(data)
        
        # 防止缓冲区过大
        if len(self.rx_buffer) > 8192:
            self.rx_buffer = self.rx_buffer[-4096:]
    
    def parse_frames(self) -> List[NodeFrame2]:
        """
        从缓冲区解析所有完整的数据帧
        
        Returns:
            解析出的数据帧列表
        """
        frames = []
        
        while True:
            frame = self._extract_one_frame()
            if frame is None:
                break
            
            parsed = self._parse_node_frame2(frame)
            if parsed is not None:
                frames.append(parsed)
        
        return frames
    
    def _extract_one_frame(self) -> Optional[bytearray]:
        """提取一个完整的数据帧"""
        # 查找帧头 0x55 0x04
        header_pos = -1
        for i in range(len(self.rx_buffer) - 1):
            if self.rx_buffer[i] == 0x55 and self.rx_buffer[i + 1] == 0x04:
                header_pos = i
                break
        
        # 如果找到帧头但不在起始位置，丢弃前面的数据
        if header_pos > 0:
            del self.rx_buffer[:header_pos]
        
        # 检查是否有足够的数据读取帧长度
        if len(self.rx_buffer) < 4:
            return None
        
        # 读取帧长度（小端序）
        frame_len = struct.unpack_from('<H', self.rx_buffer, 2)[0]
        
        # 验证帧长度
        if frame_len < self.MIN_FRAME_LENGTH or frame_len > self.MAX_FRAME_LENGTH:
            # 长度异常，丢弃一个字节继续查找
            del self.rx_buffer[0]
            return None
        
        # 检查是否接收到完整帧
        if len(self.rx_buffer) < frame_len:
            return None
        
        # 提取帧数据
        frame = bytearray(self.rx_buffer[:frame_len])
        del self.rx_buffer[:frame_len]
        
        # 校验和验证
        if not self._verify_checksum(frame):
            print("警告: 校验和失败，丢弃一帧")
            return None
        
        return frame
    
    def _verify_checksum(self, data: bytearray) -> bool:
        """验证校验和"""
        if len(data) < 2:
            return False
        
        checksum = sum(data[:-1]) & 0xFF
        return checksum == data[-1]
    
    def _parse_node_frame2(self, frame: bytearray) -> Optional[NodeFrame2]:
        """解析NodeFrame2数据帧"""
        if len(frame) < self.MIN_FRAME_LENGTH:
            return None
        
        try:
            msg = NodeFrame2(
                role=frame[4],
                id=frame[5],
                system_time_ms=struct.unpack_from('<I', frame, 6)[0],
                eop_x=frame[10] / 100.0,
                eop_y=frame[11] / 100.0,
                eop_z=frame[12] / 100.0,
                pos_x=self._read_i24_le(frame, 13) / 1000.0,
                pos_y=self._read_i24_le(frame, 16) / 1000.0,
                pos_z=self._read_i24_le(frame, 19) / 1000.0,
                vel_x=self._read_i24_le(frame, 22) / 10000.0,
                vel_y=self._read_i24_le(frame, 25) / 10000.0,
                vel_z=self._read_i24_le(frame, 28) / 10000.0,
                local_time_ms=struct.unpack_from('<I', frame, 102)[0],
                voltage_v=struct.unpack_from('<H', frame, 116)[0] / 1000.0,
                valid_node_quantity=frame[118],
                anchors=[]
            )
            
            # 计算期望的帧大小
            blocks_start = 119
            block_size = 13
            expected_size = blocks_start + block_size * msg.valid_node_quantity + 1  # +1 for checksum
            
            if len(frame) != expected_size:
                print(f"警告: 帧大小不匹配 - 实际:{len(frame)}, 期望:{expected_size}, 锚点数:{msg.valid_node_quantity}")
                return None
            
            # 解析锚点观测数据
            for i in range(msg.valid_node_quantity):
                offset = blocks_start + i * block_size
                anchor = AnchorObservation(
                    role=frame[offset + 0],
                    id=frame[offset + 1],
                    distance_m=self._read_i24_le(frame, offset + 2) / 1000.0,
                    fp_rssi_db=-2.0 * frame[offset + 5],
                    rx_rssi_db=-2.0 * frame[offset + 6]
                )
                msg.anchors.append(anchor)
            
            return msg
            
        except Exception as e:
            print(f"解析帧错误: {e}")
            return None
    
    @staticmethod
    def _read_i24_le(data: bytearray, offset: int) -> int:
        """
        读取24位有符号整数（小端序）
        按照手册推荐做法：左移到int32高位后再除以256，保持符号位
        """
        temp = (
            (data[offset] << 8) |
            (data[offset + 1] << 16) |
            (data[offset + 2] << 24)
        )
        # 转换为有符号32位整数
        if temp >= 0x80000000:
            temp -= 0x100000000
        return temp // 256
    
    def frame_to_dict(self, frame: NodeFrame2) -> Dict[str, Any]:
        """将数据帧转换为字典（用于调试）"""
        return {
            "tag_id": frame.id,
            "system_time_ms": frame.system_time_ms,
            "local_time_ms": frame.local_time_ms,
            "voltage_v": frame.voltage_v,
            "pos": [frame.pos_x, frame.pos_y, frame.pos_z],
            "vel": [frame.vel_x, frame.vel_y, frame.vel_z],
            "eop": [frame.eop_x, frame.eop_y, frame.eop_z],
            "anchors": [
                {
                    "id": a.id,
                    "distance_m": a.distance_m,
                    "fp_rssi_db": a.fp_rssi_db,
                    "rx_rssi_db": a.rx_rssi_db,
                    "delta_db": a.rx_rssi_db - a.fp_rssi_db
                }
                for a in frame.anchors
            ]
        }
