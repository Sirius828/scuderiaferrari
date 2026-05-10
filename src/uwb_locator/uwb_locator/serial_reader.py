"""
UWB串口通信模块
负责从串口读取原始数据
"""

import serial
import threading
import time
from typing import Optional, Callable


class UWBSerialReader:
    """UWB串口数据读取器"""
    
    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 921600):
        """
        初始化串口
        
        Args:
            port: 串口设备路径
            baudrate: 波特率
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_port: Optional[serial.Serial] = None
        self.running = False
        self.read_thread: Optional[threading.Thread] = None
        self.data_callback: Optional[Callable[[bytes], None]] = None
        
    def open(self) -> bool:
        """打开串口"""
        try:
            self.serial_port = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.01,  # 非阻塞读取
                write_timeout=0.01
            )
            print(f"成功打开串口: {self.port} @ {self.baudrate}")
            return True
        except Exception as e:
            print(f"打开串口失败: {e}")
            return False
    
    def close(self):
        """关闭串口"""
        self.stop()
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print(f"串口已关闭: {self.port}")
    
    def set_data_callback(self, callback: Callable[[bytes], None]):
        """设置数据接收回调函数"""
        self.data_callback = callback
    
    def start(self):
        """启动读取线程"""
        if not self.serial_port or not self.serial_port.is_open:
            if not self.open():
                return
        
        self.running = True
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()
        print("串口读取线程已启动")
    
    def stop(self):
        """停止读取线程"""
        self.running = False
        if self.read_thread:
            self.read_thread.join(timeout=2.0)
            self.read_thread = None
    
    def _read_loop(self):
        """读取循环"""
        while self.running:
            try:
                if self.serial_port and self.serial_port.in_waiting > 0:
                    data = self.serial_port.read(self.serial_port.in_waiting)
                    if data and self.data_callback:
                        self.data_callback(data)
                else:
                    time.sleep(0.001)  # 短暂休眠避免CPU占用过高
            except Exception as e:
                print(f"串口读取错误: {e}")
                time.sleep(0.1)
    
    def read_bytes(self, size: int = 1024) -> bytes:
        """直接读取指定字节数（同步方式）"""
        if not self.serial_port or not self.serial_port.is_open:
            return b""
        
        try:
            return self.serial_port.read(size)
        except Exception as e:
            print(f"读取字节失败: {e}")
            return b""
