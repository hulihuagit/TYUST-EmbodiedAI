import sys
import json
import os
import time
import math
import colorsys
import csv
from enum import Enum
from pathlib import Path
from datetime import datetime
from ctypes import *
import threading
import numpy as np
try:
    from scipy.optimize import least_squares
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
import matplotlib
from typing import List, Tuple
import copy
import asyncio

matplotlib.use('Agg')  # 适配NiceGUI的非交互式后端
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import logging
from logging.handlers import RotatingFileHandler

PROGRAM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROGRAM_DIR.parent
RAW_DATA_DIR = PROJECT_ROOT / "二、原始磁触觉数据集"

# 导入NiceGUI核心组件
from nicegui import ui, app, events
import plotly.graph_objects as go
import plotly.express as px

# -------------------------- 基础配置与常量定义 --------------------------
MAX_PATH = 260  # Windows系统路径长度限制
# 日志系统配置
logger = logging.getLogger('MagneticFieldLogger')
logger.setLevel(logging.DEBUG)
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

# 文件日志处理器（10MB轮转，保留5个备份）
file_handler = RotatingFileHandler(
    log_dir / f"magnetic_measurement_{datetime.now().strftime('%Y%m%d')}.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)
# 控制台日志处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
# 日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# -------------------------- 硬件相关类定义 --------------------------
class DEV_INFOR(Structure):
    """CH347设备信息结构体"""
    _fields_ = [
        ("iIndex", c_ubyte),
        ("DevicePath", c_ubyte * MAX_PATH),
        ("UsbClass", c_ubyte),
        ("FuncType", c_ubyte),
        ("DeviceID", c_char * 64),
        ("ChipMode", c_ubyte),
        ("DevHandle", c_void_p),
        ("BulkOutEndpMaxSize", c_ushort),
        ("BulkInEndpMaxSize", c_ushort),
        ("UsbSpeedType", c_ubyte),
        ("CH347IfNum", c_ubyte),
        ("DataUpEndp", c_ubyte),
        ("DataDnEndp", c_ubyte),
        ("ProductString", c_char * 64),
        ("ManufacturerString", c_char * 64),
        ("WriteTimeout", c_ulong),
        ("ReadTimeout", c_ulong),
        ("FuncDescStr", c_char * 64),
        ("FirewareVer", c_ubyte)
    ]


class MeasurementResult:
    """测量结果数据类（含系统时间戳）"""

    def __init__(self):
        self.T: float = 0.0  # 温度(°C)
        self.X: float = 0.0  # X轴磁场(μT)
        self.Y: float = 0.0  # Y轴磁场(μT)
        self.Z: float = 0.0  # Z轴磁场(μT)
        self.total: float = 0.0  # 总磁场强度(μT)
        self.timestamp: float = 0.0  # 相对时间戳(秒)
        self.system_time: str = ""  # 系统时间（精确到毫秒）

    def reset(self) -> None:
        """重置测量结果"""
        self.T = 0.0
        self.X = 0.0
        self.Y = 0.0
        self.Z = 0.0
        self.total = 0.0
        self.timestamp = 0.0
        self.system_time = ""

    def __str__(self) -> str:
        return (f"{self.system_time} - T={self.T:.2f}°C, X={self.X:.2f}μT, Y={self.Y:.2f}μT, "
                f"Z={self.Z:.2f}μT, Total={self.total:.2f}μT")


class I2C_MODE(Enum):
    """I2C通信速率枚举"""
    speed_20k = 0
    speed_100k = 1
    speed_400k = 2
    speed_750k = 3
    speed_50k = 4
    speed_200k = 5
    speed_1M = 6


class DeviceError(Exception):
    """设备相关异常"""
    pass


class CommunicationError(DeviceError):
    """I2C通信异常"""
    pass


class CH347Device:
    """CH347 USB转I2C设备控制类"""

    def __init__(self, usb_dev: int = 0):
        self.usb_id: int = usb_dev
        self.dev_info: DEV_INFOR = DEV_INFOR()
        self.openflag: bool = False
        self.ch347 = None
        try:
            # 加载CH347驱动（支持多路径查找）
            dll_paths = [str(PROGRAM_DIR / "CH347DLLA64.dll"), "C:/Windows/System32/CH347DLLA64.dll"]
            for path in dll_paths:
                try:
                    self.ch347 = windll.LoadLibrary(path)
                    logger.debug(f"成功加载CH347驱动: {path}")
                    break
                except:
                    continue
            if not self.ch347:
                raise DeviceError("未找到CH347DLLA64.dll驱动文件")

            if self.ch347.CH347OpenDevice(self.usb_id) != -1:
                logger.info("USB CH347设备打开成功!")
                self.openflag = True
            else:
                raise DeviceError("USB CH347设备打开失败")
        except Exception as e:
            logger.error(f"CH347初始化失败: {str(e)}")
            self.openflag = False
            if self.ch347:
                try:
                    self.ch347.CH347CloseDevice(self.usb_id)
                except:
                    pass

    def get_dev_info(self) -> None:
        """获取设备信息"""
        if not self.openflag:
            raise DeviceError("设备未打开，无法获取信息")
        try:
            self.ch347.CH347GetDeviceInfor(self.usb_id, byref(self.dev_info))
            logger.debug(
                f"CH347设备信息: 固件版本={self.dev_info.FirewareVer}, 产品名={self.dev_info.ProductString.decode()}")
        except Exception as e:
            logger.error(f"获取设备信息失败: {str(e)}")
            raise

    def close(self) -> None:
        """关闭设备"""
        if self.openflag and self.ch347:
            try:
                self.ch347.CH347CloseDevice(self.usb_id)
                self.openflag = False
                logger.info("CH347设备已关闭")
            except Exception as e:
                logger.error(f"关闭CH347设备失败: {str(e)}")


class MLX90393:
    """MLX90393磁场传感器控制类"""

    def __init__(self, ch347_device: CH347Device, i2c_dev: int = 0x0f, debug: bool = False):
        self.ch347 = ch347_device.ch347
        self.usb_id = ch347_device.usb_id
        self.openflag = ch347_device.openflag
        self.debug = debug
        self.dev_addr = i2c_dev  # 传感器I2C地址
        self.resolution_xy = 0.15  # XY轴分辨率(μT/LSB)
        self.resolution_z = 0.242  # Z轴分辨率(μT/LSB)
        self.stray_field = {"X": 0.0, "Y": 0.0, "Z": 0.0, "total": 0.0}  # 杂散磁场基准
        self.has_stray_field = False  # 是否已校准杂散磁场
        self.i2c_set()  # 初始化I2C

    def i2c_set(self) -> None:
        """配置I2C速率（默认400k，匹配measure5.py）"""
        if not self.openflag:
            raise DeviceError("CH347设备未打开")
        try:
            if self.ch347.CH347I2C_Set(self.usb_id, I2C_MODE.speed_400k.value) != 1:
                raise CommunicationError("I2C速率配置失败")
            logger.debug("I2C速率配置为400k")
        except Exception as e:
            logger.error(f"I2C配置失败: {str(e)}")
            raise

    def start_poll_measurement(self, retries: int = 3) -> None:
        """启动单次测量（带重试机制）"""
        if not self.openflag:
            raise DeviceError("CH347设备未打开")
        t_buffer = (c_byte * 2)()
        i_buffer = (c_byte * 1)()
        t_buffer[0] = self.dev_addr << 1  # 写地址
        t_buffer[1] = 0x3f  # 启动测量命令

        for attempt in range(retries):
            try:
                # 发送测量命令
                if self.ch347.CH347StreamI2C(self.usb_id, 2, t_buffer, 0, i_buffer) != 1:
                    raise CommunicationError(f"发送命令失败（尝试{attempt + 1}/{retries}）")
                # 读取响应
                t_buffer[0] = self.dev_addr << 1 | 0x01  # 读地址
                if self.ch347.CH347StreamI2C(self.usb_id, 1, t_buffer, 1, i_buffer) != 1:
                    raise CommunicationError(f"接收响应失败（尝试{attempt + 1}/{retries}）")
                # 检查状态
                if not self.check_status(i_buffer[0]):
                    raise CommunicationError(f"状态检查失败（尝试{attempt + 1}/{retries}）")
                if self.debug:
                    logger.debug("测量启动成功")
                return
            except CommunicationError as e:
                logger.warning(f"测量启动失败: {str(e)}")
                if attempt == retries - 1:
                    raise
                time.sleep(0.01)  # 重试间隔

    def read_measurement(self, retries: int = 3) -> MeasurementResult:
        """读取测量结果（带重试机制）"""
        if not self.openflag:
            raise DeviceError("CH347设备未打开")
        t_buffer = (c_byte * 2)()
        i_buffer = (c_byte * 9)()
        t_buffer[0] = self.dev_addr << 1  # 写地址
        t_buffer[1] = 0x4F  # 读取测量命令

        for attempt in range(retries):
            try:
                # 发送读取命令
                if self.ch347.CH347StreamI2C(self.usb_id, 2, t_buffer, 0, i_buffer) != 1:
                    raise CommunicationError(f"发送读取命令失败（尝试{attempt + 1}/{retries}）")
                # 读取9字节数据
                t_buffer[0] = self.dev_addr << 1 | 0x01  # 读地址
                if self.ch347.CH347StreamI2C(self.usb_id, 1, t_buffer, 9, i_buffer) != 1:
                    raise CommunicationError(f"接收数据失败（尝试{attempt + 1}/{retries}）")
                # 检查状态
                if not self.check_status(i_buffer[0]):
                    raise CommunicationError(f"状态检查失败（尝试{attempt + 1}/{retries}）")

                # 解析原始数据
                result = MeasurementResult()
                result_bytes = bytes(i_buffer)
                result.T = (result_bytes[1] << 8) + result_bytes[2]  # 原始温度值
                result.X = int.from_bytes(result_bytes[3:5], byteorder='big', signed=True)  # X轴原始值
                result.Y = int.from_bytes(result_bytes[5:7], byteorder='big', signed=True)  # Y轴原始值
                result.Z = int.from_bytes(result_bytes[7:9], byteorder='big', signed=True)  # Z轴原始值

                if self.debug:
                    logger.debug(f"原始数据: T={result.T}, X={result.X}, Y={result.Y}, Z={result.Z}")
                return result
            except CommunicationError as e:
                logger.warning(f"读取结果失败: {str(e)}")
                if attempt == retries - 1:
                    raise
                time.sleep(0.01)
        raise CommunicationError("所有重试均失败，无法读取测量结果")

    def check_status(self, status: int) -> bool:
        """检查传感器状态（0x10为命令错误）"""
        if status & 0x10:
            logger.error("传感器返回命令错误（Command Error）")
            return False
        return True

    def get_stray_field(self, samples: int = 10, interval: float = 0.1) -> bool:
        """获取杂散磁场基准值（与measure.py一致）"""
        if not self.openflag:
            logger.error("CH347设备未打开，无法校准")
            return False
        logger.info(f"正在获取杂散磁场基准值，将采集{samples}个样本...")
        x_values: list[float] = []
        y_values: list[float] = []
        z_values: list[float] = []

        for i in range(samples):
            try:
                self.start_poll_measurement()
                time.sleep(interval)
                raw_result = self.read_measurement()
                if raw_result is None:
                    logger.warning(f"获取基准值失败（第{i + 1}/{samples}次）")
                    continue
                converted = self.convert_result(raw_result, subtract=False)
                x_values.append(converted.X)
                y_values.append(converted.Y)
                z_values.append(converted.Z)
                logger.info(
                    f"基准值采样 {i + 1}/{samples}: X={converted.X:.2f}μT, Y={converted.Y:.2f}μT, Z={converted.Z:.2f}μT, Total={converted.total:.2f}μT")
                time.sleep(interval)
            except Exception as e:
                logger.error(f"校准样本{i + 1}失败: {str(e)}")
                continue

        if not x_values:
            logger.error("无法获取足够的基准值样本!")
            return False

        self.stray_field["X"] = sum(x_values) / len(x_values)
        self.stray_field["Y"] = sum(y_values) / len(y_values)
        self.stray_field["Z"] = sum(z_values) / len(z_values)
        self.stray_field["total"] = math.sqrt(
            self.stray_field["X"] ** 2 + self.stray_field["Y"] ** 2 + self.stray_field["Z"] ** 2)
        self.has_stray_field = True
        logger.info(
            f"杂散磁场基准值已获取: X={self.stray_field['X']:.2f}μT, Y={self.stray_field['Y']:.2f}μT, Z={self.stray_field['Z']:.2f}μT, Total={self.stray_field['total']:.2f}μT")
        return True

    def get_rotation_calibration(self, samples: int = 36, interval: float = 0.2, out_dir: str | None = None) -> dict:
        """基于旋转采样的校准：
        - 采集 `samples` 次测量，假设在此期间用户手动旋转传感器以平均方向依赖项。
        - 计算采样 X/Y/Z 的均值作为杂散场（self.stray_field），并把 has_stray_field 设为 True。
        - 把每一帧的原始寄存器、转换后未扣除值、以及扣除均值后的校准值保存为 CSV（如果提供 out_dir，则写入文件）。

        返回值：包含 'stray_field' 与 'samples'（列表）的字典，samples 中每项为 dict。
        """
        if not self.openflag:
            logger.error("CH347设备未打开，无法校准（旋转方式）")
            return {"success": False, "reason": "device_closed"}

        samples_list: list[dict] = []
        x_vals: list[float] = []
        y_vals: list[float] = []
        z_vals: list[float] = []

        logger.info(f"开始旋转采样校准（{samples} 次，间隔 {interval}s），请在采样期间缓慢旋转传感器")
        for i in range(samples):
            try:
                self.start_poll_measurement()
                time.sleep(interval)
                raw_result = self.read_measurement()
                if raw_result is None:
                    logger.warning(f"旋转采样第 {i+1}/{samples} 读取失败")
                    continue
                conv = self.convert_result(raw_result, subtract=False)
                # 记录转换后（未扣除）值与原始寄存器
                rec = {
                    "index": i,
                    "timestamp": datetime.now().isoformat(),
                    # 使用 convert_result 中保存的原始寄存器值 conv.T_raw（先前误用 conv.T）
                    "T_raw": getattr(conv, 'T_raw', None),
                    "T_C": getattr(conv, 'T', None),
                    "X_raw": getattr(conv, 'X_raw', None),
                    "Y_raw": getattr(conv, 'Y_raw', None),
                    "Z_raw": getattr(conv, 'Z_raw', None),
                    "X_uT": conv.X,
                    "Y_uT": conv.Y,
                    "Z_uT": conv.Z,
                    "total_uT": getattr(conv, 'total', math.sqrt(conv.X**2 + conv.Y**2 + conv.Z**2)),
                }
                samples_list.append(rec)
                x_vals.append(conv.X)
                y_vals.append(conv.Y)
                z_vals.append(conv.Z)
                logger.debug(f"旋转采样 {i+1}/{samples}: X={conv.X:.2f} Y={conv.Y:.2f} Z={conv.Z:.2f} T={conv.T:.2f}")
            except Exception as e:
                logger.warning(f"旋转采样第 {i+1} 次失败: {e}")
                time.sleep(interval)
                continue

        if not x_vals:
            logger.error("未能获取到任何有效旋转采样")
            return {"success": False, "reason": "no_samples"}

        mean_x = sum(x_vals) / len(x_vals)
        mean_y = sum(y_vals) / len(y_vals)
        mean_z = sum(z_vals) / len(z_vals)
        mean_total = math.sqrt(mean_x ** 2 + mean_y ** 2 + mean_z ** 2)

        # 设置为杂散场基准并开启扣除开关
        self.stray_field["X"] = mean_x
        self.stray_field["Y"] = mean_y
        self.stray_field["Z"] = mean_z
        self.stray_field["total"] = mean_total
        self.has_stray_field = True

        # 计算每一帧的校准后值（扣除均值）并准备写入
        for rec in samples_list:
            rec["X_cal_uT"] = rec["X_uT"] - mean_x
            rec["Y_cal_uT"] = rec["Y_uT"] - mean_y
            rec["Z_cal_uT"] = rec["Z_uT"] - mean_z
            rec["total_cal_uT"] = math.sqrt(rec["X_cal_uT"]**2 + rec["Y_cal_uT"]**2 + rec["Z_cal_uT"]**2)

        # 写 CSV（如果指定目录或默认目录 results/）
        try:
            out_dir = out_dir or os.path.join(os.getcwd(), 'results')
            os.makedirs(out_dir, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            fname = f"rotation_cal_{hex(self.dev_addr)}_{ts}.csv"
            out_path = os.path.join(out_dir, fname)
            with open(out_path, 'w', newline='', encoding='utf-8') as csvf:
                writer = csv.writer(csvf)
                header = [
                    'index', 'timestamp', 'T_raw', 'T_C',
                    'X_raw', 'Y_raw', 'Z_raw',
                    'X_uT', 'Y_uT', 'Z_uT', 'total_uT',
                    'X_cal_uT', 'Y_cal_uT', 'Z_cal_uT', 'total_cal_uT'
                ]
                writer.writerow(header)
                for r in samples_list:
                    writer.writerow([
                        r.get('index'), r.get('timestamp'), r.get('T_raw'), r.get('T_C'),
                        r.get('X_raw'), r.get('Y_raw'), r.get('Z_raw'),
                        r.get('X_uT'), r.get('Y_uT'), r.get('Z_uT'), r.get('total_uT'),
                        r.get('X_cal_uT'), r.get('Y_cal_uT'), r.get('Z_cal_uT'), r.get('total_cal_uT'),
                    ])
            logger.info(f"旋转校准采样已保存: {out_path}")
        except Exception as e:
            logger.warning(f"保存旋转校准 CSV 失败: {e}")
            out_path = None

        return {
            "success": True,
            "stray_field": {"X": mean_x, "Y": mean_y, "Z": mean_z, "total": mean_total},
            "samples": samples_list,
            "csv": out_path,
        }

    def convert_result(self, result: MeasurementResult, subtract: bool = True) -> MeasurementResult:
        """转换测量结果并扣除杂散磁场（与measure.py一致）"""
        # 保存原始寄存器值以便调试（result.T 初始为原始寄存器读取值）
        raw_T_reg = result.T
        # 暂存原始寄存器到结果对象，供后续直接访问原始数据
        try:
            result.T_raw = int(raw_T_reg)
        except Exception:
            result.T_raw = raw_T_reg
        # 转换为摄氏度：线性公式（来自设备/实验标定）
        result.T = (result.T - 46244) / 45.2 + 25
        # 记录原始寄存器值与转换后的温度，便于定位各传感器温差来源（debug）
        try:
            logger.debug(f"MLX90393@0x{self.dev_addr:02x} raw_T_reg={raw_T_reg} -> temp={result.T:.2f}C")
        except Exception:
            pass
        # 保存原始寄存器的磁场计数值，供记录/调试使用
        try:
            result.X_raw = int(result.X)
        except Exception:
            result.X_raw = result.X
        try:
            result.Y_raw = int(result.Y)
        except Exception:
            result.Y_raw = result.Y
        try:
            result.Z_raw = int(result.Z)
        except Exception:
            result.Z_raw = result.Z

        # 把原始寄存器值转换为 μT
        result.X = result.X_raw * self.resolution_xy
        result.Y = result.Y_raw * self.resolution_xy
        result.Z = result.Z_raw * self.resolution_z
        if subtract and self.has_stray_field:
            result.X -= self.stray_field["X"]
            result.Y -= self.stray_field["Y"]
            result.Z -= self.stray_field["Z"]
        result.total = math.sqrt(result.X ** 2 + result.Y ** 2 + result.Z ** 2)
        return result


# -------------------------- NiceGUI界面核心类 --------------------------
class MagneticMeasurementUI:
    """磁场测量可视化系统UI（匹配图片布局）

    物理传感器编号说明（根据板上连接器朝向与照片观察）：

        连接器在 PCB 底部（图中朝向为底部），传感器按矩阵排列为 2x2：

          S1    S2   <-- PCB 上方
          [U1]  [U2]

          S3    S4   <-- PCB 下方（靠近连接器一侧）
          [U3]  [U4]

    因此默认代码中 S1..S4 的物理位置为：
    - S1: PCB 左上（top-left）
    - S2: PCB 右上（top-right）
    - S3: PCB 左下（bottom-left）
    - S4: PCB 右下（bottom-right)

    说明：实际 I2C 地址到 S1..S4 的对应关系以构造 `MagneticMeasurementUI` 时传入的
    `sensor_addresses` 顺序为准（按 S1,S2,S3,S4 的物理顺序）。若想在 `config.json`
    中指定地址顺序，可添加键 `sensor_address_sequence`，值为地址列表（整数或十六进制字符串），
    程序会优先使用该顺序重排传感器实例。
    """

    def __init__(self, ch347_device: CH347Device, sensor_addresses: List[int]):
        self.ch347 = ch347_device  # CH347设备实例
        # 先加载配置（允许用户在 config.json 中指定地址顺序映射）
        self.config = self.load_config()

        # sensor_addresses 参数通常是一个 int 列表（例如 [0x0c,0x0d,0x0e,0x0f]）
        # 若 config.json 中提供了 "sensor_address_sequence"，则按其顺序重排。
        seq = self.config.get('sensor_address_sequence', None)
        if seq:
            normalized: List[int] = []
            for v in seq:
                try:
                    if isinstance(v, str):
                        addr = int(v, 16) if v.lower().startswith('0x') else int(v)
                    else:
                        addr = int(v)
                except Exception:
                    continue
                if addr in sensor_addresses and addr not in normalized:
                    normalized.append(addr)
            # 把未在 config 中列出的地址追加到末尾，保证所有传入地址被保留
            for a in sensor_addresses:
                if a not in normalized:
                    normalized.append(a)
            self.sensor_addresses = normalized
        else:
            self.sensor_addresses = sensor_addresses  # 传感器I2C地址列表

        # 按 self.sensor_addresses 创建传感器实例（顺序对应 S1..S4）
        self.sensors = [MLX90393(ch347_device, addr) for addr in self.sensor_addresses]  # 传感器实例列表
        self.running = False  # 测量状态标记
        self.paused = False  # 暂停状态标记
        self.pause_start_time = 0  # 暂停开始时间
        self.total_pause_duration = 0  # 总暂停时长
        self.measurement_data = {  # 测量数据存储（按传感器地址分组）
            hex(addr): {"times": [], "system_times": [], "Ts": [], "T_raws": [], "Xs": [], "Ys": [], "Zs": [], "totals": []}
            for addr in self.sensor_addresses
        }
        self.plot_update_interval = 0.1  # 图表更新间隔（秒）
        self.last_plot_time = 0  # 上次图表更新时间
        self.sensor_settling_delay = max(0.0025, float(self.config.get("sensor_settling_delay", 0.004)))
        self.sensor_read_retries = max(1, int(self.config.get("sensor_read_retries", 2)))
        self.sensor_outlier_floor = max(0.0, float(self.config.get("sensor_outlier_floor", 180.0)))
        self.sensor_outlier_multiplier = max(1.0, float(self.config.get("sensor_outlier_multiplier", 6.0)))
        self.sensor_total_limit = max(0.0, float(self.config.get("sensor_total_limit", 9000.0)))
        self.last_valid_results = {hex(addr): None for addr in self.sensor_addresses}
        self.sensor_spike_counters = {hex(addr): 0 for addr in self.sensor_addresses}
        self.current_sensor = "0xc"  # 当前显示的传感器（保留但不作为默认显示）
        self.current_view_mode = "compare"  # 默认显示模式改为对比模式
        self.current_data_type = "total"  # 默认数据类型为总磁场
        # 多选对比选择（支持 X/Y/Z/温度/总场），可在单图上显示多条曲线
        self.compare_selections: set = set()
        # 复选框 UI 占位属性（将在 build_ui 中赋值）
        self.cb_x = None
        self.cb_y = None
        self.cb_z = None
        self.cb_temp = None
        self.cb_total = None
        # 相对显示：以开始测量时的第一帧为基线（与measure方案保持绝对值，默认关闭）
        self.relative_display = False
        self.baseline = {
            'x': {},
            'y': {},
            'z': {},
            'total': {}
        }
        # 3D可视化相关属性
        self.history_data = []  # 用于存储临时数据
        self.sensor_addr_map = {hex(addr): index for index, addr in enumerate(self.sensor_addresses)}
        self.sensor_hex_sequence = [hex(addr) for addr in self.sensor_addresses]
        sensor_count = len(self.sensor_addresses)

        # 温度偏置（可用于校准各传感器之间的恒定差异），从配置读取或初始化为 0
        temp_offsets_cfg = self.config.get('temp_offsets', {}) if isinstance(self.config, dict) else {}
        try:
            self.temp_offsets = {hex(addr): float(temp_offsets_cfg.get(hex(addr), temp_offsets_cfg.get(str(hex(addr)), 0.0))) for addr in self.sensor_addresses}
        except Exception:
            self.temp_offsets = {hex(addr): 0.0 for addr in self.sensor_addresses}

        # 支持从 config.json 中读取 PCB 与传感器的实际尺寸（单位 mm），否则使用保守默认值
        pcb_size = self.config.get('pcb_size_mm', [34.0, 26.0])
        try:
            pcb_w = float(pcb_size[0])
            pcb_h = float(pcb_size[1])
        except Exception:
            pcb_w, pcb_h = 34.0, 26.0

        # 默认传感器坐标（当配置中未提供时使用基于实际PCB的近似值）
        sensors_cfg = self.config.get('sensor_positions', None)

        if sensor_count > 0:
            # 使用照片测量得到的常见布局（以PCB左下角为参考点的近似坐标，单位 mm）
            # 示例：S1(7,18), S2(14,18), S3(7,11), S4(14,11) —— 高度 z 取传感器封装厚度约 1.2mm
            sensor_layout = np.array([
                [7.0, 18.0, 1.2],
                [14.0, 18.0, 1.2],
                [7.0, 11.0, 1.2],
                [14.0, 11.0, 1.2],
            ], dtype=float)
        else:
            # 圆形布局（备选）
            radius = max(pcb_w, pcb_h) / 2.0 + 6.0
            angles = np.linspace(0.0, 2 * math.pi, sensor_count, endpoint=False)
            xy_positions = np.array(
                [[radius * math.cos(angle), radius * math.sin(angle)] for angle in angles],
                dtype=float
            )
            # 传感器默认高度取 1.2mm
            sensor_layout = np.column_stack(
                (xy_positions, np.full(sensor_count, 1.2, dtype=float))
            )

        # 为后续可视化使用，中心化坐标系：将 PCB 中心视为 (0,0)
        # 先把 sensor_layout 当作相对 PCB 左下角的坐标（如果用户提供的是中心坐标则此处不会破坏）
        # 尝试检测是否坐标已经是中心化（若均值接近0则认为已中心化）
        mean_xy = np.mean(sensor_layout[:, :2], axis=0) if sensor_layout.size else np.array([0.0, 0.0])
        if abs(mean_xy[0]) > 1e-3 and abs(mean_xy[1]) > 1e-3 and (max(abs(mean_xy[0]), abs(mean_xy[1])) > max(pcb_w, pcb_h) * 0.1):
            # 如果坐标明显不是以中心为原点，则把其从左下角坐标转换为以中心为原点
            sensor_layout[:, 0] = sensor_layout[:, 0] - pcb_w / 2.0
            sensor_layout[:, 1] = sensor_layout[:, 1] - pcb_h / 2.0

        self.sensor_plane_height = float(np.mean(sensor_layout[:, 2]) if sensor_layout.size else 1.2)
        self.sensor_positions = sensor_layout

        # 如果配置中存在已固定的最大采样频率，则加载并显示
        try:
            self.max_sample_frequency = float(self.config.get('max_sample_frequency')) if self.config.get('max_sample_frequency') is not None else None
        except Exception:
            self.max_sample_frequency = None

        # 根据PCB尺寸设置板子显示范围与高度（保留足够的边距）
        self.assembly_extent = max(pcb_w, pcb_h) / 2.0 + 20.0
        self.board_extent = self.assembly_extent
        # 板高度取略低于传感器的 z（用于将传感器显示在板上方）
        self.board_height = min(self.sensor_plane_height, 2.0) - 1.5

        # 仅移除“检测磁体”功能：保留三维展示必须的基础属性。
        # 若后续需要重新添加，可在此处恢复更复杂的磁场/偶极拟合逻辑。
        self.magnet_positions = np.zeros((0, 3), dtype=float)  # 空磁源占位
        self.field_projection = 'orthographic'  # 三维图相机投影模式需要此属性

        self.sensor_indicator_offset = 2.8
        self.field_cmin = 0.0
        self.field_cmax = 80.0
        # 显示用的传感器横向放大倍数（不改变实际测量坐标，仅影响可视化距离感）
        self.sensor_display_scale = 1.4
        # 是否显示传感器处的磁感方向箭头（交互可控）
        self.show_direction_cones = True
        # 相关可视化功能已移除（相关配置已清理）
        # 方向箭头的大小参考值（可由UI滑块调整）
        # 提高默认尺寸以便在深色界面中更容易看到（用户可通过UI滑块继续调整）
        self.direction_sizeref = 24.0
        self.sensor_colorscale = [
            [0.0, '#06102b'],
            [0.18, '#0f52ba'],
            [0.36, '#0ea5e9'],
            [0.54, '#22d3ee'],
            [0.72, '#38bdf8'],
            [0.88, '#34d399'],
            [1.0, '#22c55e'],
        ]
        # 相关 trace 名称与索引已移除
        self._dir_cone_index = None
        self._sensor_mesh_indices = []
        self._sensor_label_index = None
        self._magnet_marker_index = None
        self.board_extent = 70.0
        self.board_height = -5.0
        # 偶极子自动拟合功能已移除（相关逻辑与 UI 控件均已删除）
        # 占位：不再维护 auto fit 的运行时变量
        self.playback_running = False
        self.playback_timer: threading.Timer | None = None
        self.playback_index = 0
        self.playback_total_frames = 0
        self.playback_speed = 0.12
        self.playback_btn = None

        # 界面组件存储（后续动态更新用）
        self.status_label = None
        self.output_text = None
        self.progress_bar = None
        self.magnetic_max_display = None
        self.magnetic_min_display = None
    # (已移除帧内平均显示控件占位)

        # 每个传感器的累计平均显示（按传感器顺序 S1..S4）
        self.sensor_avg_displays = []

        # 数据访问锁，保护 self.measurement_data 的并发读写
        self.data_lock = threading.Lock()

        self.plot_elements = {}  # 图表元素存储
        self.three_d_plot = None  # 三维磁场图存储
        # 输出窗口元素ID用于JS自动滚动
        self.output_text_id = f"output_text_{int(time.time() * 1000)}"
        # 四传感器独立图表
        self.sensor_plots: list = []
        # 保存对话框默认值：每次成功保存后更新，下一次打开对话框时沿用
        self.last_save_defaults: dict | None = self._normalize_save_defaults(
            self.config.get('last_save_defaults') if isinstance(self.config, dict) else None
        )

    def _normalize_save_defaults(self, defaults: dict | None) -> dict | None:
        """归一化保存对话框默认值：category/force/index。"""
        if not isinstance(defaults, dict):
            return None
        try:
            category = str(defaults.get('category', '')).strip() or 'Sieve'
            force = str(defaults.get('force', '')).strip() or '10'
            index = max(1, int(defaults.get('index', 1)))
            return {
                'category': category,
                'force': force,
                'index': index,
            }
        except Exception:
            return None

    def _persist_last_save_defaults(self) -> None:
        """把上一次成功保存的默认值持久化到 config.json。"""
        if not self.last_save_defaults:
            return
        normalized = self._normalize_save_defaults(self.last_save_defaults)
        if not normalized:
            return
        try:
            cfg = {}
            if os.path.exists('config.json'):
                with open('config.json', 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        cfg = loaded
            cfg['last_save_defaults'] = normalized
            with open('config.json', 'w', encoding='utf-8') as f:
                json.dump(cfg, f, ensure_ascii=False, indent=4)
            if isinstance(self.config, dict):
                self.config['last_save_defaults'] = normalized
        except Exception as e:
            logger.warning(f"持久化 last_save_defaults 失败: {e}")

    @staticmethod
    def _normalize_category_lookup_key(value: str) -> str:
        """Normalize category text for directory matching."""
        return ''.join(ch for ch in str(value).lower() if ch.isalnum())

    def _get_builtin_save_category_dirs(self) -> dict[str, str]:
        """Preferred directory names for built-in save categories."""
        return {
            'gluestick': '01_GlueStick_data',
            'phone': '02_Phone_data',
            'case': '03_Case_data',
            'ppball': '04_PpBall_data',
            'crystal': '05_Crystal_data',
            'rugby': '06_Rugby_data',
            'sieve': '07_Sieve_data',
            'charger': '08_Charger_data',
            'shaizi': '09_shaizi_data',
            'mouse': '10_Mouse_data',
            'poweradapter': '11_PowerAdapter_data',
        }

    def load_config(self) -> dict:
        """加载配置文件（采样间隔、测量时长等）"""
        config = {
            "sample_interval": 0.05,
            "measurement_duration": 20,
            "log_level": "DEBUG",
            "sensor_settling_delay": 0.004,
            "sensor_read_retries": 2,
            "sensor_outlier_floor": 180.0,
            "sensor_outlier_multiplier": 6.0,
            "sensor_total_limit": 9000.0,
        }
        try:
            if os.path.exists('config.json'):
                with open('config.json', 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    if isinstance(file_config, dict):
                        config.update(file_config)
        except Exception as e:
            logger.warning(f"加载配置失败: {str(e)}")
        return config

    def save_config(self) -> None:
        """保存配置文件"""
        config = {
            "sample_frequency": self.sample_frequency.value if getattr(self, 'sample_frequency', None) is not None else (1.0 / self.config.get('sample_interval', 0.05)),
            "measurement_duration": self.measurement_duration.value,
            "log_level": self.log_level.value,
            "sensor_settling_delay": self.sensor_settling_delay,
            "sensor_read_retries": self.sensor_read_retries,
            "sensor_outlier_floor": self.sensor_outlier_floor,
            "sensor_outlier_multiplier": self.sensor_outlier_multiplier,
            "sensor_total_limit": self.sensor_total_limit,
        }
        # 保存温度偏置（按 hex 地址字符串）
        try:
            config['temp_offsets'] = {addr: float(v) for addr, v in getattr(self, 'temp_offsets', {}).items()}
        except Exception:
            pass
        try:
            if getattr(self, 'max_sample_frequency', None) is not None:
                config['max_sample_frequency'] = float(self.max_sample_frequency)
        except Exception:
            pass
        try:
            normalized = self._normalize_save_defaults(getattr(self, 'last_save_defaults', None))
            if normalized:
                config['last_save_defaults'] = normalized
        except Exception:
            pass
        try:
            with open('config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            self.add_output(f"配置已保存到 config.json")
        except Exception as e:
            self.add_output(f"保存配置失败: {str(e)}", level="ERROR")

    def add_output(self, message: str, level: str = "INFO") -> None:
        """添加输出信息到日志窗口，并写入日志系统"""
        # 日志系统记录
        log_func = {
            "DEBUG": logger.debug,
            "INFO": logger.info,
            "WARNING": logger.warning,
            "ERROR": logger.error
        }.get(level, logger.info)
        log_func(message)
        # UI输出窗口显示（带时间戳）
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.output_text.value += f"[{timestamp}] [{level}] {message}\n"
        # 自动滚动在UI线程的定时器中统一执行，避免后台线程创建UI元素

    def update_log_level(self, e: events.ValueChangeEventArguments) -> None:
        """更新日志级别"""
        level = e.value
        level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}
        console_handler.setLevel(level_map.get(level, logging.INFO))
        self.add_output(f"日志级别已设置为: {level}")

    def calibrate_stray_field(self) -> None:
        """校准杂散磁场（多线程执行，避免UI冻结）"""
        if self.running:
            self.add_output("测量正在进行中，无法校准", level="WARNING")
            return

        def calibration_task():
            self.add_output("开始校准杂散磁场...")
            for sensor in self.sensors:
                addr = hex(sensor.dev_addr)
                self.add_output(f"正在校准传感器 {addr}...")
                try:
                    success = sensor.get_stray_field(samples=10, interval=0.1)
                    if success:
                        self.add_output(f"传感器 {addr} 校准完成")
                    else:
                        self.add_output(f"传感器 {addr} 校准失败", level="WARNING")
                except Exception as e:
                    self.add_output(f"传感器 {addr} 校准出错: {str(e)}", level="ERROR")
            self.add_output("杂散磁场校准流程结束")

        threading.Thread(target=calibration_task, daemon=True).start()

    def calibrate_stray_field_rotation(self, samples: int = 36, interval: float = 0.2, out_dir: str | None = None) -> None:
        """基于旋转采样的杂散场校准入口（后台线程）。
        会对每个传感器调用 `MLX90393.get_rotation_calibration` 并把结果路径输出到 UI 日志。
        """
        if self.running:
            self.add_output("测量正在进行中，无法校准（旋转方式）", level="WARNING")
            return

        def task():
            self.add_output(f"开始基于旋转的杂散场校准：每传感器采样 {samples} 次，间隔 {interval}s。请开始缓慢旋转设备。")
            if not self.sensors:
                self.add_output("未检测到传感器，校准终止", level="ERROR")
                return
            for sensor in self.sensors:
                addr = hex(sensor.dev_addr)
                self.add_output(f"对传感器 {addr} 执行旋转采样校准...")
                try:
                    res = sensor.get_rotation_calibration(samples=samples, interval=interval, out_dir=out_dir)
                    if not res or not res.get('success', False):
                        self.add_output(f"传感器 {addr} 旋转校准失败: {res.get('reason') if isinstance(res, dict) else res}", level="WARNING")
                    else:
                        sf = res.get('stray_field', {})
                        csvp = res.get('csv')
                        self.add_output(f"传感器 {addr} 旋转校准完成: stray_field X={sf.get('X',0):.2f}μT Y={sf.get('Y',0):.2f}μT Z={sf.get('Z',0):.2f}μT, CSV={csvp}")
                except Exception as e:
                    self.add_output(f"传感器 {addr} 旋转校准出错: {e}", level="ERROR")
            self.add_output("旋转校准流程结束")

        threading.Thread(target=task, daemon=True).start()

    # 已移除：按中位数校准温度功能（由用户请求删除）

    

    def measure_max_sampling_frequency_rigorous(self, test_duration: float = 8.0, warmup: float = 1.0) -> None:
        """对整个传感器阵列进行严格测试，测量在连续读取情况下每次完整采样集合的时长，
        输出稳健统计并把保守值写入 self.max_sample_frequency 与 config.json（持久化）。

        算法：先做短时热机（warmup），然后在 test_duration 时间内反复执行完整的传感器循环，
        记录每次循环耗时，计算每次循环频率的统计量（中位数、10百分位等），并选择保守值。
        """
        if self.running:
            self.add_output("正在测量中，无法进行严格频率测试", level="WARNING")
            return

        def task():
            try:
                self.add_output("开始严格测试最大采样频率：热机中...")
                if not self.sensors:
                    self.add_output("未检测到传感器，测试终止", level="ERROR")
                    return
                # 简短热机期，避免冷启动导致的首次耗时偏大
                end_warm = time.time() + warmup
                while time.time() < end_warm:
                    for sensor in self.sensors:
                        try:
                            sensor.start_poll_measurement()
                            time.sleep(self.sensor_settling_delay)
                            _ = sensor.read_measurement()
                        except Exception:
                            # 忽略单次错误
                            time.sleep(0.005)
                            continue

                self.add_output(f"开始正式测试，持续 {test_duration:.1f}s ...")
                cycle_times: list[float] = []
                test_start = time.time()
                # 记录每次完整循环的开始-结束时间
                while time.time() - test_start < test_duration:
                    cycle_start = time.time()
                    for sensor in self.sensors:
                        try:
                            sensor.start_poll_measurement()
                            time.sleep(self.sensor_settling_delay)
                            _ = sensor.read_measurement()
                        except Exception:
                            # 若单次失败，不中断循环，继续下一个传感器
                            time.sleep(0.005)
                            continue
                    cycle_end = time.time()
                    cycle_times.append(cycle_end - cycle_start)

                if not cycle_times:
                    self.add_output("未能获取到有效循环时间，测试失败", level="ERROR")
                    return

                # 计算每次循环对应的频率
                freqs = [1.0 / t if t > 0 else 0.0 for t in cycle_times]
                median_f = float(np.median(freqs))
                p10 = float(np.percentile(freqs, 10))
                min_f = float(min(freqs))
                # 选择保守值：取 10% 分位与中位数*0.9 的较小者
                conservative = min(p10, median_f * 0.9)
                # 保证不会返回极端高值
                conservative = max(0.1, conservative)

                # 写入并持久化
                self.max_sample_frequency = conservative
                try:
                    self.max_freq_label.text = f"(固定最大: {conservative:.2f} Hz)"
                except Exception:
                    pass
                self.config['max_sample_frequency'] = float(conservative)
                self.save_config()
                # 不自动覆盖用户输入的采样频率（保留测试结果供用户查看或手动应用）
                # 但弹出确认对话框，询问用户是否将该固定值应用到采样频率
                try:
                    self._pending_max_freq = conservative
                except Exception:
                    self._pending_max_freq = conservative
                try:
                    self.apply_dialog_label.text = f"严格测试完成，建议最大采样频率 {conservative:.2f} Hz。是否应用到当前采样频率？"
                except Exception:
                    pass
                try:
                    ui.timer(0, lambda: self.apply_confirm_dialog.open(), once=True)
                except Exception:
                    pass
                # 额外后备：在客户端弹出浏览器原生确认框（如果NiceGUI对话框未能显示时仍可操作）
                try:
                    # 使用字符串格式化避免 f-string 中的花括号冲突
                    js = ("if(window.confirm('严格测试完成，建议最大采样频率 %0.2f Hz。是否应用？')) { var el = document.getElementById(\'sample_frequency_input\'); "
                          "if (el) { el.value = '%0.2f'; el.dispatchEvent(new Event('input', {bubbles:true})); el.dispatchEvent(new Event('change', {bubbles:true})); } }") % (conservative, round(conservative, 2))
                    ui.timer(0, lambda js=js: ui.run_javascript(js), once=True)
                except Exception:
                    pass
                self.add_output(f"严格测试完成：median={median_f:.2f}Hz, p10={p10:.2f}Hz, min={min_f:.2f}Hz; 固定值={conservative:.2f}Hz")
            except Exception as e:
                self.add_output(f"严格测试出错: {str(e)}", level="ERROR")

        threading.Thread(target=task, daemon=True).start()

    def calibrate_temp_to_reference(self, reference_temp: float = 25.0, sample_count: int = 10) -> None:
        """根据参考温度对每个传感器计算偏置，使其平均值接近参考温度。"""
        if self.running:
            self.add_output("测量正在进行中，无法校准温度偏置", level="WARNING")
            return

        def task():
            self.add_output(f"开始按参考温度 {reference_temp}°C 校准...")
            with self.data_lock:
                avgs = {}
                for addr in self.sensor_hex_sequence:
                    data = self.measurement_data.get(addr, {}).get('Ts', [])
                    if not data:
                        continue
                    slice_vals = data[-sample_count:]
                    try:
                        avgs[addr] = float(sum(slice_vals) / len(slice_vals))
                    except Exception:
                        continue

            if not avgs:
                self.add_output("没有足够的温度样本进行校准", level="WARNING")
                return

            for addr, avg in avgs.items():
                new_offset = reference_temp - avg
                self.temp_offsets[addr] = new_offset
                self.add_output(f"设置 {addr} 偏置 = {new_offset:+.2f}°C（原均值 {avg:.2f} -> 目标 {reference_temp:.2f}）")

            try:
                self.config['temp_offsets'] = {addr: float(v) for addr, v in self.temp_offsets.items()}
                self.save_config()
            except Exception:
                pass
            self.add_output("按参考温度校准完成")

        threading.Thread(target=task, daemon=True).start()

    def _apply_pending_max_freq(self) -> None:
        """把待应用的最大频率写入采样频率输入框（用户在确认对话框中点击应用时调用）。"""
        try:
            val = float(getattr(self, '_pending_max_freq', None))
        except Exception:
            val = None
        if val:
            try:
                self.sample_frequency.value = round(val, 2)
                self.add_output(f"已将采样频率设置为 {val:.2f} Hz")
            except Exception:
                self.add_output("应用采样频率失败", level="ERROR")
        try:
            # 关闭对话框
            self.apply_confirm_dialog.close()
        except Exception:
            pass

    def start_measurement(self) -> None:
        """开始测量（多线程执行）"""
        if self.running:
            self.add_output("测量已在进行中", level="WARNING")
            return

        self.stop_playback()

        # 验证参数（采样频率改为输入频率，转换为采样间隔）
        try:
            # 不要在这里创建 UI 控件作为默认值（会在页面上重复渲染多个输入控件）
            if not hasattr(self, 'sample_frequency') or getattr(self, 'sample_frequency', None) is None:
                ui.notify('采样频率控件不存在，请重启或刷新页面', type='error')
                return
            # 读取用户输入的频率和时长
            freq = float(self.sample_frequency.value)
            duration = float(self.measurement_duration.value)
            if freq <= 0:
                ui.notify("采样频率必须大于0", type="error")
                return
            if duration <= 0:
                ui.notify("测量时长必须大于0", type="error")
                return
            # 若已测试最大采样频率，检查是否超出
            maxf = getattr(self, 'max_sample_frequency', None)
            if maxf and freq > maxf:
                ui.notify('已超出系统最大采样频率，请重新输入', type='error')
                return
            interval = 1.0 / freq
        except Exception as e:
            ui.notify(f"参数错误: {str(e)}", type="error")
            return

        self.running = True
        self.paused = False
        self.pause_start_time = 0
        self.total_pause_duration = 0
        self.last_plot_update_time = time.time()
        # 更新测量状态
        self.measure_status.text = "运行中"
        self.measure_status.classes("text-success font-medium text-sm")

        # 更新按钮状态
        self.start_measure_btn.disable()
        self.pause_measure_btn.enable()
        self.resume_measure_btn.disable()
        self.stop_measure_btn.enable()
        self.calibrate_btn.disable()
        if self.playback_btn:
            self.playback_btn.disable()

        # 开始独立的三维图更新定时器 (20Hz更新频率)
        if hasattr(self, 'plot_update_timer') and self.plot_update_timer:
            self.plot_update_timer.cancel()

        def update_3d_plot_periodically():
            if self.running and not self.paused:
                try:
                    self.update_3d_magnetic_field_data()
                except Exception as e:
                    logger.warning(f"更新三维图失败: {str(e)}")
                # 继续定时更新
                self.plot_update_timer = threading.Timer(0.05, update_3d_plot_periodically)
                self.plot_update_timer.daemon = True
                self.plot_update_timer.start()

        # 启动定时器
        self.plot_update_timer = threading.Timer(0.05, update_3d_plot_periodically)
        self.plot_update_timer.daemon = True
        self.plot_update_timer.start()
        # 重置数据（加锁以防三维/UI线程并发访问）
        with self.data_lock:
            for addr in self.measurement_data:
                self.measurement_data[addr] = {"times": [], "system_times": [], "Ts": [], "T_raws": [], "Xs": [], "Ys": [], "Zs": [],
                                               "totals": []}
        # 重置相对显示基线
        self.baseline = {'x': {}, 'y': {}, 'z': {}, 'total': {}}
        # 重置进度条（NiceGUI进度条值范围是0-1）
        self.progress_bar.value = 0.0
        self.add_output(f"开始测量 - 间隔={interval}s, 时长={duration}s")

        def measurement_task():
            # 在测量任务开始时重置暂停相关变量
            self.paused = False
            self.pause_start_time = 0
            self.total_pause_duration = 0

            start_time = time.time()
            sample_count = 0
            expected_samples = int(duration / interval)  # 预期样本数
            self.add_output(f"开始测量 - 间隔={interval}s, 时长={duration}s, 预期样本数={expected_samples}")

            # 采用固定频率采样，忽略延迟累积
            next_sample_time = start_time
            elapsed = 0  # 初始化实际经过时间

            while self.running and elapsed < duration:
                current_time = time.time()

                # 检查是否暂停
                if self.paused:
                    time.sleep(0.1)  # 暂停时短暂休眠，等待继续或停止
                    continue

                # 计算实际经过时间（只在非暂停状态下计算）
                current_pause_duration = self.total_pause_duration
                elapsed = current_time - start_time - current_pause_duration

                # 确保 elapsed 不为负数
                if elapsed < 0:
                    elapsed = 0

                # 检查是否到达采样时间点（基于实际时间）
                expected_sample_time = sample_count * interval
                if elapsed >= expected_sample_time:
                    sample_count += 1

                    # 更新进度（基于实际测量时间）
                    progress = (elapsed / duration) * 100
                    # 确保进度在合理范围内
                    if progress < 0:
                        progress = 0
                    elif progress > 100:
                        progress = 100

                    # 设置进度条值（NiceGUI的进度条值范围是0-1，不是0-100）
                    progress_value = progress / 100.0
                    # 直接设置进度条值
                    self.progress_bar.value = progress_value
                    self.status_label.text = f"测量中 | 已采样{sample_count}次 | 进度{progress:.1f}% | 实际时间{elapsed:.1f}s | 暂停{current_pause_duration:.1f}s"

                    # 读取所有传感器数据（并行处理以提高效率）
                    system_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    sensor_data = {}

                    # 快速读取所有传感器
                    for sensor in self.sensors:
                        addr = hex(sensor.dev_addr)
                        converted_data = self._read_sensor_with_validation(sensor, elapsed, system_time)
                        if converted_data is not None:
                            sensor_data[addr] = converted_data

                    # 批量存储数据（减少重复操作），对 measurement_data 写入加锁
                    with self.data_lock:
                        for addr, converted_data in sensor_data.items():
                            self.measurement_data[addr]["times"].append(elapsed)
                            self.measurement_data[addr]["system_times"].append(system_time)
                            # 存储未经加工的原始温度寄存器值（T_raw）以及转换后的摄氏度（Ts）
                            try:
                                raw_val = getattr(converted_data, 'T_raw', None)
                            except Exception:
                                raw_val = None
                            self.measurement_data[addr]["T_raws"].append(raw_val)
                            # 保持 Ts 为转换后的摄氏度（未自动应用 temp_offsets），以便 UI 兼容显示
                            self.measurement_data[addr]["Ts"].append(converted_data.T)
                            self.measurement_data[addr]["Xs"].append(converted_data.X)
                            self.measurement_data[addr]["Ys"].append(converted_data.Y)
                            self.measurement_data[addr]["Zs"].append(converted_data.Z)
                            self.measurement_data[addr]["totals"].append(converted_data.total)

                            # 初始化基线（仅首次为每个地址设置）
                            if addr not in self.baseline['x']:
                                self.baseline['x'][addr] = converted_data.X
                                self.baseline['y'][addr] = converted_data.Y
                                self.baseline['z'][addr] = converted_data.Z
                                self.baseline['total'][addr] = converted_data.total

                    # 更新磁场极值显示（使用第一个有效传感器的数据）
                    if sensor_data:
                        totals = [data.total for data in sensor_data.values()]
                        if totals:
                            max_total_value = max(totals)
                            min_total_value = min(totals)
                            try:
                                recorded_max = float(self.magnetic_max_display.text)
                            except (TypeError, ValueError):
                                recorded_max = None
                            try:
                                recorded_min = float(self.magnetic_min_display.text)
                            except (TypeError, ValueError):
                                recorded_min = None
                            if recorded_max is None or max_total_value > recorded_max:
                                self.magnetic_max_display.text = f"{max_total_value:.2f}"
                            if recorded_min is None or min_total_value < recorded_min:
                                self.magnetic_min_display.text = f"{min_total_value:.2f}"

                        # 帧内平均显示已移除（不再更新平均 X/Y/Z/总场的单独标签）

                        # 计算每个传感器从测量开始到当前的累计平均（总场），并更新对应显示（读取加锁）
                        try:
                            with self.data_lock:
                                for idx, addr in enumerate(self.sensor_hex_sequence):
                                    if idx < len(self.sensor_avg_displays):
                                        data = self.measurement_data.get(addr)
                                        if data and data.get('totals'):
                                            vals = data['totals']
                                            if vals:
                                                avg_val = sum(vals) / len(vals)
                                                self.sensor_avg_displays[idx].text = f"{avg_val:.2f}"
                                            else:
                                                self.sensor_avg_displays[idx].text = "-"
                                        else:
                                            self.sensor_avg_displays[idx].text = "-"
                        except Exception:
                            # 忽略更新错误，不影响测量
                            pass

                    # 每10次采样记录一次详细日志
                    if sample_count % 10 == 0 and sensor_data:
                        first_addr, first_data = list(sensor_data.items())[0]
                        self.add_output(f"采样{sample_count}: 传感器{first_addr}: {first_data}")

                    # 计算下一个采样时间点（基于实际时间）
                    # 下一次采样时间 = 开始时间 + 采样次数 * 间隔 + 总暂停时间

                    # 每次采样都更新图表，确保三维磁场可视化的实时性
                    self.update_plots()
                    self.last_plot_time = current_time

                # 短暂休眠，减少CPU占用
                time.sleep(0.001)  # 1ms休眠

            # 测量结束处理
            self.running = False
            self.paused = False
            self.pause_start_time = 0
            self.total_pause_duration = 0
            self.progress_bar.value = 1.0  # 100%完成
            self.status_label.text = f"测量完成 | 共采样{sample_count}次"
            # 更新测量状态
            self.measure_status.text = "已完成"
            self.measure_status.classes("text-accent font-medium text-sm")

            self.start_measure_btn.enable()
            self.pause_measure_btn.disable()
            self.resume_measure_btn.disable()
            self.stop_measure_btn.disable()
            self.calibrate_btn.enable()
            if self.playback_btn:
                self.playback_btn.enable()
            self.update_plots()  # 最后更新一次图表
            # 最终计算并显示每个传感器的累计平均总场（读取加锁）
            try:
                with self.data_lock:
                    for idx, addr in enumerate(self.sensor_hex_sequence):
                        if idx < len(self.sensor_avg_displays):
                            data = self.measurement_data.get(addr)
                            if data and data.get('totals'):
                                vals = data['totals']
                                if vals:
                                    avg_val = sum(vals) / len(vals)
                                    self.sensor_avg_displays[idx].text = f"{avg_val:.2f}"
                                else:
                                    self.sensor_avg_displays[idx].text = "-"
                            else:
                                self.sensor_avg_displays[idx].text = "-"
            except Exception:
                pass

            self.add_output(f"测量结束，共采集{sample_count}个有效样本")

        threading.Thread(target=measurement_task, daemon=True).start()

    def stop_measurement(self) -> None:
        """停止测量"""
        if not self.running:
            return
        self.running = False
        self.paused = False
        self.pause_start_time = 0
        self.total_pause_duration = 0
        self.stop_playback()

        # 取消三维图更新定时器
        if hasattr(self, 'plot_update_timer') and self.plot_update_timer:
            self.plot_update_timer.cancel()
            self.plot_update_timer = None

        # 重置进度条到初始状态
        self.progress_bar.value = 0.0
        self.add_output("正在停止测量...")

        # 重置按钮状态
        self.start_measure_btn.enable()
        self.pause_measure_btn.disable()
        self.resume_measure_btn.disable()
        self.stop_measure_btn.disable()
        self.calibrate_btn.enable()

        # 更新状态显示
        self.measure_status.text = "已停止"
        self.measure_status.classes("text-muted font-medium text-sm")
        self.status_label.text = "准备就绪"  # 重置为初始状态
        if self.playback_btn:
            self.playback_btn.enable()

    def pause_measurement(self) -> None:
        """暂停测量"""
        if not self.running or self.paused:
            return
        self.paused = True
        self.pause_start_time = time.time()  # 记录暂停开始时间
        self.add_output("测量已暂停")
        # 更新按钮状态
        self.pause_measure_btn.disable()
        self.resume_measure_btn.enable()
        # 更新状态显示
        self.measure_status.text = "已暂停"
        self.measure_status.classes("text-warning font-medium text-sm")
        self.status_label.text += " | 已暂停"

    def resume_measurement(self) -> None:
        """继续测量"""
        if not self.running or not self.paused:
            return

        # 计算暂停时长并累积
        pause_duration = 0
        if self.pause_start_time > 0:
            pause_duration = time.time() - self.pause_start_time
            self.total_pause_duration += pause_duration

        self.paused = False
        self.pause_start_time = 0
        self.add_output(f"测量已继续（本次暂停{pause_duration:.2f}秒，总暂停{self.total_pause_duration:.2f}秒）")
        # 更新按钮状态
        self.pause_measure_btn.enable()
        self.resume_measure_btn.disable()
        # 更新状态显示
        self.measure_status.text = "运行中"
        self.measure_status.classes("text-success font-medium text-sm")
        # 更新状态标签（移除暂停标识）
        current_text = self.status_label.text
    def toggle_playback(self) -> None:
        """切换三维回放状态"""
        if self.playback_running:
            self.stop_playback(user_triggered=True)
        else:
            self.start_playback()

    def start_playback(self) -> None:
        """开始三维历史数据回放"""
        if self.running:
            self.add_output("测量进行中，无法回放", level="WARNING")
            ui.notify("请先停止实时测量后再回放", type="warning")
            return

        with self.data_lock:
            frame_counts = [len(dataset["times"]) for dataset in self.measurement_data.values()]
        total_frames = max(frame_counts) if frame_counts else 0
        if total_frames <= 0:
            self.add_output("暂无历史数据可回放", level="WARNING")
            ui.notify("暂无可回放的测量数据", type="warning")
            return

        self.stop_playback()
        self.playback_running = True
        self.playback_index = 0
        self.playback_total_frames = total_frames

        if self.playback_btn:
            self.playback_btn.text = "停止回放"
            self.playback_btn.classes("btn-warning px-4 py-2 text-sm")

        self.add_output(f"开始三维回放，共 {total_frames} 帧")
        self._schedule_playback_step()

    def _schedule_playback_step(self) -> None:
        if not self.playback_running:
            return

        frame_data = self._get_frame_snapshot(self.playback_index)
        if frame_data:
            self.update_3d_magnetic_field_data(frame_data=frame_data, advance_phase=True)

        self.playback_index += 1
        if self.playback_index >= self.playback_total_frames:
            self.stop_playback(finished=True)
            return

        self.playback_timer = threading.Timer(self.playback_speed, self._schedule_playback_step)
        self.playback_timer.daemon = True
        self.playback_timer.start()

    def _get_frame_snapshot(self, index: int) -> dict[str, dict[str, float]]:
        snapshot: dict[str, dict[str, float]] = {}
        # 读取 measurement_data 做快照以保证线程安全
        with self.data_lock:
            for addr_hex, dataset in self.measurement_data.items():
                if len(dataset["times"]) > index:
                    snapshot[addr_hex] = {
                        "X": dataset["Xs"][index],
                        "Y": dataset["Ys"][index],
                        "Z": dataset["Zs"][index],
                        "total": dataset["totals"][index],
                        "T": dataset["Ts"][index],
                        "timestamp": dataset["times"][index],
                        "system_time": dataset["system_times"][index] if len(dataset["system_times"]) > index else ""
                    }
        return snapshot

    def stop_playback(self, finished: bool = False, user_triggered: bool = False) -> None:
        if self.playback_timer:
            self.playback_timer.cancel()
            self.playback_timer = None

        was_running = self.playback_running
        self.playback_running = False
        self.playback_index = 0
        self.playback_total_frames = 0

        if self.playback_btn:
            self.playback_btn.text = "回放测量过程"
            self.playback_btn.classes("btn-secondary px-4 py-2 text-sm")
            if not self.running:
                self.playback_btn.enable()

        if was_running:
            message = "三维回放完成" if finished else "三维回放已停止"
            self.add_output(message)
            if finished:
                ui.notify("三维回放结束", type="positive")

        current_text = self.status_label.text if self.status_label else ""
        if current_text and " | 已暂停" in current_text:
            self.status_label.text = current_text.replace(" | 已暂停", "")

    def clear_output(self) -> None:
        """清空输出窗口"""
        self.output_text.value = ""
        self.add_output("输出窗口已清空")

    def save_measurement_data(self) -> None:
        """保存测量数据到CSV文件"""
        if not any(len(data["times"]) > 0 for data in self.measurement_data.values()):
            ui.notify("无有效测量数据可保存", type="warning")
            return

        # 打开自定义文件名对话框
        self.show_save_dialog()

    def show_save_dialog(self) -> None:
        """显示保存文件名对话框：按顺序输入 类别、力值、序号，保存时组合为 类别_力值N_序号（序号两位）"""
        # 生成默认文件名（示例格式：GlueStick_10N_01）；若已有上次成功保存值，则优先沿用
        if self.last_save_defaults:
            default_name = (
                f"{self.last_save_defaults.get('category', 'Sieve')}_"
                f"{self.last_save_defaults.get('force', '10')}N_"
                f"{int(self.last_save_defaults.get('index', 1)):02d}"
            )
        else:
            default_name = "Sieve_10N_01"
        default_cat = ""
        default_force = ""
        default_index = ""
        try:
            parts = default_name.split('_')
            if len(parts) >= 1:
                default_cat = parts[0]
            if len(parts) >= 2:
                f = parts[1]
                default_force = f[:-1] if f.lower().endswith('n') else f
            if len(parts) >= 3:
                default_index = parts[2]
        except Exception:
            pass

        # 类别下拉选项：优先从 data_new 目录解析已有类别，失败时使用固定兜底
        category_options = []
        try:
            data_new_root = RAW_DATA_DIR
            if data_new_root.exists() and data_new_root.is_dir():
                for child in sorted(data_new_root.iterdir()):
                    if not child.is_dir():
                        continue
                    name = child.name
                    # 目录名形如: 01_GlueStick_data
                    if '_' in name and name.lower().endswith('_data'):
                        parts = name.split('_')
                        if len(parts) >= 3:
                            category_options.append(parts[1])
        except Exception:
            category_options = []

        builtin_category_options = [
            'GlueStick', 'Phone', 'Case', 'PpBall', 'Crystal', 'Rugby', 'Sieve', 'charger',
            'shaizi',
            'Mouse', 'PowerAdapter',
        ]

        if not category_options:
            category_options = builtin_category_options.copy()
        else:
            category_options.extend(builtin_category_options)

        # 去重并保持顺序
        category_options = list(dict.fromkeys(category_options))
        default_category = default_cat if default_cat in category_options else category_options[0]

        with ui.dialog() as dialog, ui.card().style('min-width: 420px'):
            with ui.column().classes('gap-4 p-4'):
                ui.label('保存测量数据').classes('text-lg font-bold text-primary')
                ui.label('按顺序输入：类别、力值（N），序号（两位）').classes('text-sm text-secondary')

                # 三个输入：类别（下拉）、力值、序号
                category_input = ui.select(
                    options=category_options,
                    value=default_category,
                    label='类别'
                ).classes('w-full').props('outlined')
                force_input = ui.input(placeholder='例如：10（表示10N）', value=default_force).classes('w-full').props('outlined')

                # 序号改为带加减的步进控件（默认 01 -> 数值 1）
                try:
                    default_index_value = int(default_index)
                    if default_index_value < 1:
                        default_index_value = 1
                except Exception:
                    default_index_value = 1

                def _change_index(delta: int) -> None:
                    try:
                        current = int(index_input.value or 1)
                    except Exception:
                        current = 1
                    next_value = max(1, current + delta)
                    index_input.value = next_value

                with ui.row().classes('items-center w-full gap-2'):
                    ui.label('序号').classes('text-sm text-secondary')
                    ui.button('−1', on_click=lambda: _change_index(-1)).classes('btn-secondary px-3 py-1')
                    index_input = ui.number(value=default_index_value, min=1, step=1).classes('w-24').props('outlined')
                    ui.button('+1', on_click=lambda: _change_index(1)).classes('btn-secondary px-3 py-1')

                ui.label('说明：系统将生成一个包含所有传感器数据的CSV文件，文件名形如 GlueStick_10N_01').classes('text-xs text-muted')

                with ui.row().classes('gap-2 justify-end w-full mt-4'):
                    ui.button('取消', on_click=dialog.close).classes('btn-secondary px-4 py-2')
                    ui.button(
                        '保存',
                        on_click=lambda: self.confirm_save(
                            dialog,
                            category_input.value,
                            force_input.value,
                            str(int(index_input.value or 1))
                        )
                    ).classes('btn-primary px-4 py-2')

        dialog.open()

    def confirm_save(self, dialog, category: str, force: str, index: str) -> None:
        """确认保存并执行保存操作：接收三个输入，组合成安全的文件名后保存"""
        dialog.close()

        # 基本验证
        if not category or not category.strip():
            ui.notify("类别不能为空", type="warning")
            return
        if not force or not str(force).strip():
            ui.notify("力值不能为空", type="warning")
            return
        if not index or not str(index).strip():
            ui.notify("序号不能为空", type="warning")
            return

        # 清理并格式化各字段
        import re
        cat_clean = re.sub(r'[<>:"/\\|?*\s]+', '_', category.strip())
        force_raw = str(force).strip()
        # 去掉可能输入的单位 N 或 n
        if force_raw.lower().endswith('n'):
            force_raw = force_raw[:-1]
        force_clean = re.sub(r'[^0-9\.-]', '', force_raw)
        if force_clean == '':
            ui.notify("力值格式不正确，请输入数字，例如 10", type="warning")
            return

        # 序号格式化为两位，例如 1 -> 01
        try:
            idx_int = int(str(index).strip())
            if idx_int < 0:
                raise ValueError()
            idx_formatted = f"{idx_int:02d}"
        except Exception:
            # 如果用户传入已经是两位形式（例如 '01'），尝试直接使用数字部分
            num_match = re.search(r'\d+', str(index))
            if num_match:
                try:
                    idx_int = int(num_match.group(0))
                    idx_formatted = f"{idx_int:02d}"
                except Exception:
                    ui.notify("序号格式不正确", type="warning")
                    return
            else:
                ui.notify("序号格式不正确", type="warning")
                return

        # 组合文件名：类别_力值N_序号
        combined = f"{cat_clean}_{force_clean}N_{idx_formatted}"
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', combined)

        # 选择目标目录：优先匹配 data_new 下包含类别名的子文件夹（不区分大小写、忽略非字母数字）
        def _norm(s: str) -> str:
            return self._normalize_category_lookup_key(s)

        target_dir = None
        data_new_root = RAW_DATA_DIR
        cat_norm = _norm(cat_clean)
        try:
            if data_new_root.exists() and data_new_root.is_dir():
                for child in sorted(data_new_root.iterdir()):
                    if child.is_dir():
                        name_norm = _norm(child.name)
                        if cat_norm and cat_norm in name_norm:
                            target_dir = child
                            break
        except Exception:
            target_dir = None

        if target_dir is None and cat_norm:
            builtin_dir_name = self._get_builtin_save_category_dirs().get(cat_norm)
            if builtin_dir_name:
                target_dir = data_new_root / builtin_dir_name

        # 如果没找到匹配目录，则使用旧的 data/ 目录
        if target_dir is None:
            target_dir = Path('data')

        target_dir.mkdir(parents=True, exist_ok=True)

        # 同名冲突检查：存在则先弹框确认是否强制覆盖
        candidate_path = target_dir / f"{safe_name}.csv"
        save_defaults = {
            "category": cat_clean,
            "force": force_clean,
            "index": idx_int,
        }

        if candidate_path.exists():
            ui.notify("文件已存在，请确认是否覆盖", type="warning")
            with ui.dialog() as overwrite_dialog, ui.card().style('min-width: 420px'):
                with ui.column().classes('gap-3 p-4'):
                    ui.label('文件已存在，是否强制覆盖？').classes('text-lg font-bold text-warning')
                    ui.label(str(candidate_path)).classes('text-xs text-muted break-all')
                    with ui.row().classes('gap-2 justify-end w-full mt-2'):
                        ui.button('取消', on_click=lambda: (overwrite_dialog.close(), ui.notify('已取消保存', type='warning'))).classes('btn-secondary px-4 py-2')
                        ui.button(
                            '强制覆盖',
                            on_click=lambda: self._confirm_overwrite_and_save(
                                overwrite_dialog,
                                str(target_dir),
                                safe_name,
                                save_defaults,
                            )
                        ).classes('btn-primary px-4 py-2')
            # 保留引用并延迟打开，避免与前一个dialog关闭时机冲突导致不显示
            self._overwrite_confirm_dialog = overwrite_dialog
            ui.timer(0, lambda: self._overwrite_confirm_dialog.open(), once=True)
            return

        # 无冲突直接保存
        self._save_data_task(str(target_dir), safe_name, overwrite=False, save_defaults=save_defaults)

    def _confirm_overwrite_and_save(self, dialog, dir_path: str, custom_name: str, save_defaults: dict | None = None) -> None:
        """用户确认覆盖后执行保存。"""
        try:
            dialog.close()
        except Exception:
            pass
        self._save_data_task(dir_path, custom_name, overwrite=True, save_defaults=save_defaults)

    def _save_data_task(self, dir_path: str, custom_name: str = None, overwrite: bool = True, save_defaults: dict | None = None) -> None:
        """实际执行数据保存（避免阻塞UI）"""
        if not dir_path:
            return

        # 使用自定义名称或默认时间戳
        if custom_name:
            base_name = custom_name
        else:
            base_name = datetime.now().strftime("magnetic_data_%Y%m%d_%H%M%S")

        try:
            # 在开始写文件前，先做一份 measurement_data 的深拷贝，避免长时间持锁
            with self.data_lock:
                data_copy = copy.deepcopy(self.measurement_data)

            # 合并所有传感器数据到一个CSV文件
            filename = f"{base_name}.csv"
            file_path = os.path.join(dir_path, filename)

            # 兜底防护：调用方要求不覆盖时，存在同名文件则直接返回
            if (not overwrite) and os.path.exists(file_path):
                ui.notify("文件已存在，已取消保存", type="warning")
                self.add_output(f"检测到同名文件，已取消保存: {file_path}", level="WARNING")
                return

            # 获取所有传感器数据的拷贝，按物理顺序（self.sensor_hex_sequence）确保顺序一致
            sensor_addrs = list(self.sensor_hex_sequence)
            if not sensor_addrs or not any(len(data_copy[addr]["times"]) > 0 for addr in sensor_addrs):
                self.add_output("没有有效数据可保存", level="WARNING")
                return

            # 写文件并逐行写入数据（包含原始寄存器 T_raw）
            # 写入包含原始寄存器值的表头（每个传感器增加 T_raw 列）
            header_parts = ["时间", "相对时间(s)"]
            for i in range(len(sensor_addrs)):
                s = i + 1
                header_parts.extend([
                    f"S{s}_Traw",
                    f"S{s}温度",
                    f"S{s}_X",
                    f"S{s}_Y",
                    f"S{s}_Z",
                    f"S{s}总场",
                ])

            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                f.write(",".join(header_parts) + "\n")

                # 找到最长的数据序列长度
                max_length = max(len(data_copy[addr]["times"]) for addr in sensor_addrs
                                 if len(data_copy[addr]["times"]) > 0)

                # 逐行写入数据
                for i in range(max_length):
                    row_data = []

                    # 添加时间信息（使用第一个有数据的传感器的时间）
                    time_added = False
                    for addr in sensor_addrs:
                        if len(self.measurement_data[addr]["times"]) > i:
                            if not time_added:
                                row_data.append(self.measurement_data[addr]["system_times"][i])
                                row_data.append(f"{self.measurement_data[addr]['times'][i]:.4f}")
                                time_added = True
                            break

                    if not time_added:
                        row_data.extend(["", ""])

                    # 添加每个传感器的数据（包含原始寄存器 T_raw）
                    for addr in sensor_addrs:
                        data = data_copy[addr]
                        if len(data["times"]) > i:
                            # 有数据：T_raw, 温度, X, Y, Z, 总场
                            try:
                                traw = data.get('T_raws', [None] * len(data.get('times', [])))[i]
                            except Exception:
                                traw = None
                            row_data.extend([
                                f"{traw}" if traw is not None else "",
                                f"{data['Ts'][i]:.2f}",
                                f"{data['Xs'][i]:.2f}",
                                f"{data['Ys'][i]:.2f}",
                                f"{data['Zs'][i]:.2f}",
                                f"{data['totals'][i]:.2f}"
                            ])
                        else:
                            # 无数据：填入空值
                            row_data.extend(["", "", "", "", "", ""])

                    # 写入一行数据
                    f.write(",".join(row_data) + "\n")

            self.add_output(f"所有传感器数据已合并保存: {file_path}")
            ui.notify("数据保存完成", type="success")
            self.add_output(f"数据文件已保存到 {dir_path} 目录，文件名: {filename}")
            if save_defaults:
                normalized = self._normalize_save_defaults(save_defaults)
                if normalized:
                    self.last_save_defaults = normalized
                    self._persist_last_save_defaults()

        except Exception as e:
            self.add_output(f"保存数据失败: {str(e)}", level="ERROR")
            ui.notify(f"保存失败: {str(e)}", type="error")

    def create_magnetic_plot(self) -> Figure:
        """创建磁场数据可视化图表（初始空图表）"""
        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Magnetic Field (μT)")
        ax.set_title("Real-time Magnetic Field Measurement")
        ax.grid(True, alpha=0.3)
        # 初始化曲线（每个传感器一条曲线）
        colors = ["blue", "green", "red", "purple"]
        # 按物理顺序创建曲线（保持 S1..S4 的一致性）
        for i, addr in enumerate(self.sensor_hex_sequence):
            line, = ax.plot([], [], label=f"Sensor {addr}", color=colors[i % len(colors)], linewidth=1.5)
            self.plot_elements[addr] = line
        ax.legend(loc="upper right")
        return fig

    def switch_compare_view(self, data_type: str) -> None:
        """切换到四传感器显示：仅更新按钮样式与四小图数据"""
        self.current_view_mode = "compare"
        self.current_data_type = data_type

        # 重置所有按钮样式
        self.reset_all_buttons()

        # 高亮当前按钮
        if data_type == "temp":
            self.temp_compare_btn.classes("btn-primary px-3 py-1 text-sm")
        elif data_type == "x":
            self.x_compare_btn.classes("btn-primary px-3 py-1 text-sm")
        elif data_type == "y":
            self.y_compare_btn.classes("btn-primary px-3 py-1 text-sm")
        elif data_type == "z":
            self.z_compare_btn.classes("btn-primary px-3 py-1 text-sm")
        elif data_type == "total":
            self.total_compare_btn.classes("btn-primary px-4 py-1 text-sm")

        # 刷新四个小图
        # 兼容旧调用：当单选按钮调用时，把选择集合设置为单项
        try:
            self.compare_selections = {data_type}
        except Exception:
            self.compare_selections = set()
        self.update_four_sensor_plots()

    def on_compare_checkbox_change(self, e: events.ValueChangeEventArguments | None = None) -> None:
        """处理复选框变化，维护 self.compare_selections 并刷新四小图"""
        try:
            sels = set()
            try:
                if getattr(self, 'cb_x', None) and getattr(self, 'cb_x').value:
                    sels.add('x')
                if getattr(self, 'cb_y', None) and getattr(self, 'cb_y').value:
                    sels.add('y')
                if getattr(self, 'cb_z', None) and getattr(self, 'cb_z').value:
                    sels.add('z')
                if getattr(self, 'cb_temp', None) and getattr(self, 'cb_temp').value:
                    sels.add('temp')
                if getattr(self, 'cb_total', None) and getattr(self, 'cb_total').value:
                    sels.add('total')
            except Exception:
                pass
            # 限制最多 5 项（当前只有 5 项可选）
            if len(sels) > 5:
                ui.notify('最多可选择 5 个显示项', type='warning')
                # 截断多余项
                sels = set(list(sels)[:5])
            self.compare_selections = sels
            # 若没有选择，保持当前_data_type 作为回退
            if not self.compare_selections and hasattr(self, 'current_data_type'):
                self.compare_selections = {self.current_data_type}
            self.update_four_sensor_plots()
        except Exception:
            pass

    def set_default_compare_selection(self) -> None:
        """初始化时设置默认复选框值为总场对比"""
        try:
            if getattr(self, 'cb_total', None) is not None:
                try:
                    self.cb_total.value = True
                except Exception:
                    pass
            self.on_compare_checkbox_change()
        except Exception:
            pass

    def reset_all_buttons(self) -> None:
        """重置所有按钮样式"""
        # 重置对比按钮
        self.temp_compare_btn.classes("btn-secondary px-3 py-1 text-sm")
        self.x_compare_btn.classes("btn-secondary px-3 py-1 text-sm")
        self.y_compare_btn.classes("btn-secondary px-3 py-1 text-sm")
        self.z_compare_btn.classes("btn-secondary px-3 py-1 text-sm")
        self.total_compare_btn.classes("btn-secondary px-4 py-1 text-sm")

    def update_plots(self) -> None:
        """更新磁场数据图表（简化版，显示当前选中传感器的数据）"""
        # 在NiceGUI中，直接更新UI是线程安全的，但为了避免阻塞交互，我们使用异步方式
        # 直接调用但添加异常处理
        try:
            # 更新当前传感器数据
            self.update_current_plot_data()

            # 三维图现在由独立的定时器更新，这里不再更新
        except Exception as e:
            # 捕获异常以防止UI崩溃
            logger.warning(f"更新图表失败: {str(e)}")
            # 不抛出异常，允许测量继续进行

    def update_current_plot_data(self) -> None:
        """更新四传感器小图的数据"""
        self.update_four_sensor_plots()

    def update_four_sensor_plots(self) -> None:
        """把四个传感器的数据分别填到四个小图中"""
        if len(self.sensor_plots) < 4:
            return
        data_keys = {
            'temp': 'Ts',
            'x': 'Xs',
            'y': 'Ys',
            'z': 'Zs',
            'total': 'totals'
        }
        # 支持多选：compare_selections 中的所有类型都会作为单个图的多条曲线展示
        selected = list(self.compare_selections) if getattr(self, 'compare_selections', None) else [self.current_data_type]
        if not selected:
            selected = [self.current_data_type]
        # 读取 measurement_data 时做一次快照，减少并发读写冲突
        with self.data_lock:
            # 保持物理顺序（或 config.json 中指定顺序）
            addrs = [a for a in self.sensor_hex_sequence if a in self.measurement_data]
            data_snapshot = {
                addr: {
                    'times': list(self.measurement_data[addr].get('times', [])),
                    'Ts': list(self.measurement_data[addr].get('Ts', [])),
                    'Xs': list(self.measurement_data[addr].get('Xs', [])),
                    'Ys': list(self.measurement_data[addr].get('Ys', [])),
                    'Zs': list(self.measurement_data[addr].get('Zs', [])),
                    'totals': list(self.measurement_data[addr].get('totals', [])),
                }
                for addr in addrs
            }

        # 颜色池（最多支持 5 条曲线）
        color_pool = ['#60a5fa', '#34d399', '#f97316', '#a78bfa', '#f43f5e']
        name_map = {
            'temp': '温度(°C)',
            'x': 'X轴(μT)',
            'y': 'Y轴(μT)',
            'z': 'Z轴(μT)',
            'total': '总磁场(μT)'
        }

        for i in range(4):
            plot = self.sensor_plots[i]
            if i < len(addrs):
                d = data_snapshot.get(addrs[i], {'times': [], 'totals': []})
                xs = [f"{t:.2f}" for t in d['times']]
            else:
                xs = []

            # 构建多个 series
            series_list = []
            for idx, sel in enumerate(selected[:5]):
                key = data_keys.get(sel, 'totals')
                if i < len(addrs):
                    ys = d.get(key, [])
                else:
                    ys = []
                series_list.append({
                    'type': 'line',
                    'data': ys,
                    'name': name_map.get(sel, sel),
                    'smooth': True,
                    'showSymbol': False,
                    'lineStyle': {'width': 2, 'color': color_pool[idx % len(color_pool)]}
                })

            # 更新
            try:
                plot.options.setdefault('xAxis', {})
                plot.options['xAxis']['data'] = xs
                plot.options['series'] = series_list if series_list else [{'data': []}]
                # 更新 y 轴名称为第一条的单位（若多选，则显示复数单位“多项”）
                if len(selected) == 1:
                    yname = name_map.get(selected[0], '总磁场(μT)')
                else:
                    yname = '多项比较'
                plot.options.setdefault('yAxis', {})
                plot.options['yAxis']['name'] = yname
                # 更新 legend（显示所选曲线名称）
                plot.options.setdefault('legend', {})
                plot.options['legend']['show'] = True
                plot.update()
            except Exception:
                continue

    def _is_outlier_measurement(self, addr: str, measurement: MeasurementResult) -> bool:
        """检测异常尖峰功能已取消，所有读数均视为有效"""
        return False

    def _get_fallback_measurement(
        self,
        addr: str,
        elapsed: float,
        system_time: str,
    ) -> MeasurementResult | None:
        """返回上一次有效数据作为回退"""
        last_valid = self.last_valid_results.get(addr)
        if last_valid is None:
            return None
        fallback = copy.deepcopy(last_valid)
        fallback.timestamp = elapsed
        fallback.system_time = system_time
        return fallback

    def _read_sensor_with_validation(
        self,
        sensor: MLX90393,
        elapsed: float,
        system_time: str,
    ) -> MeasurementResult | None:
        """读取传感器数据并进行尖峰过滤与回退"""
        addr = hex(sensor.dev_addr)
        latest_rejected: MeasurementResult | None = None
        relax_threshold = max(self.sensor_read_retries * 2, 6)
        for attempt in range(self.sensor_read_retries):
            try:
                sensor.start_poll_measurement()
                time.sleep(self.sensor_settling_delay)
                raw_data = sensor.read_measurement()
                converted = sensor.convert_result(raw_data)
                converted.timestamp = elapsed
                converted.system_time = system_time
                if self._is_outlier_measurement(addr, converted):
                    self.sensor_spike_counters[addr] += 1
                    latest_rejected = copy.deepcopy(converted)
                    if self.sensor_spike_counters[addr] >= relax_threshold:
                        accepted = copy.deepcopy(latest_rejected)
                        self.add_output(
                            f"传感器{addr}连续{relax_threshold}次判定异常，放宽阈值采纳当前测量（|B|={accepted.total:.2f}μT）",
                            level="WARNING"
                        )
                        self.sensor_spike_counters[addr] = 0
                        self.last_valid_results[addr] = copy.deepcopy(accepted)
                        return accepted
                    self.add_output(
                        f"传感器{addr}检测到异常尖峰（|B|={converted.total:.2f}μT），尝试重新采样",
                        level="WARNING"
                    )
                    time.sleep(self.sensor_settling_delay)
                    continue
                self.sensor_spike_counters[addr] = 0
                self.last_valid_results[addr] = copy.deepcopy(converted)
                return converted
            except Exception as exc:
                self.sensor_spike_counters[addr] += 1
                self.add_output(f"传感器{addr}读取异常: {exc}", level="WARNING")
            time.sleep(self.sensor_settling_delay)
        if latest_rejected is not None and self.sensor_spike_counters[addr] >= relax_threshold:
            accepted = copy.deepcopy(latest_rejected)
            self.add_output(
                f"传感器{addr}连续{self.sensor_spike_counters[addr]}次异常，放宽阈值采纳最近读数（|B|={accepted.total:.2f}μT）",
                level="WARNING"
            )
            self.sensor_spike_counters[addr] = 0
            self.last_valid_results[addr] = copy.deepcopy(accepted)
            return accepted
        fallback = self._get_fallback_measurement(addr, elapsed, system_time)
        if fallback is not None:
            self.add_output(
                f"传感器{addr}连续{self.sensor_spike_counters[addr]}次异常，使用上一帧有效数据回退",
                level="WARNING"
            )
        return fallback

    def create_3d_magnetic_field_plot(self, include_sensor_labels: bool = True, set_instance_attrs: bool = True) -> go.Figure:
        """创建立体风格的三维磁场可视化图表，根据实际设备照片调整布局

        参数:
        - include_sensor_labels: 是否在图上绘制传感器的 `S1..S4` 文本标签。
            默认 True。将上方面板创建为不显示标签时传入 False。
        - set_instance_attrs: 若为 False，则不修改实例跟踪的 trace 索引（用于次要面板）。
        """
        fig = go.Figure()

        board_extent = self.board_extent
        board_height = self.board_height

        x = np.linspace(-board_extent, board_extent, 2)
        y = np.linspace(-board_extent, board_extent, 2)
        grid_x, grid_y = np.meshgrid(x, y)
        grid_z = np.full_like(grid_x, board_height)

        fig.add_trace(go.Surface(
            x=grid_x,
            y=grid_y,
            z=grid_z,
            colorscale=[[0, '#111827'], [1, '#0f172a']],
            name='底板',
            showscale=False,
            opacity=0.95,
            hoverinfo='skip'
        ))

        frame_x = [board_extent, -board_extent, -board_extent, board_extent, board_extent]
        frame_y = [board_extent, board_extent, -board_extent, -board_extent, board_extent]
        frame_z = [board_height + 0.6] * len(frame_x)
        fig.add_trace(go.Scatter3d(
            x=frame_x,
            y=frame_y,
            z=frame_z,
            mode='lines',
            line=dict(color='#94a3b8', width=3),
            name='底板边框',
            hoverinfo='skip'
        ))

        # 连接器方向箭头（已移除）。保留属性占位以避免引用错误。
        if set_instance_attrs:
            try:
                self._connector_cone_index = None
                self._connector_label_index = None
            except Exception:
                self._connector_cone_index = None
                self._connector_label_index = None

        sensor_hover = []
        if self.sensor_positions.size > 0:
            self._sensor_mesh_indices = []
            cube_half = 3.6
            label_positions = []
            # 计算用于显示的传感器位置（横向放大以获得更明显的立体感），不改变原始 self.sensor_positions
            try:
                display_positions = np.asarray(self.sensor_positions, dtype=float).copy()
                center_xy = np.mean(self.sensor_positions[:, :2], axis=0)
                display_positions[:, :2] = center_xy + (self.sensor_positions[:, :2] - center_xy) * float(getattr(self, 'sensor_display_scale', 1.0))
            except Exception:
                display_positions = np.asarray(self.sensor_positions, dtype=float)

            for idx, position in enumerate(display_positions):
                center = np.array([position[0], position[1], position[2] + 4.5], dtype=float)
                label_positions.append(center + np.array([0.0, 0.0, cube_half + 1.8]))
                offsets = cube_half * np.array([
                    [-1, -1, -1],
                    [1, -1, -1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [1, 1, 1],
                    [-1, 1, 1],
                ], dtype=float)
                vertices = center + offsets
                triangles = [
                    (0, 1, 2), (0, 2, 3),
                    (4, 5, 6), (4, 6, 7),
                    (0, 1, 5), (0, 5, 4),
                    (1, 2, 6), (1, 6, 5),
                    (2, 3, 7), (2, 7, 6),
                    (3, 0, 4), (3, 4, 7),
                ]
                mesh_trace = go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=[tri[0] for tri in triangles],
                    j=[tri[1] for tri in triangles],
                    k=[tri[2] for tri in triangles],
                    intensity=[0.0] * len(vertices),
                    colorscale=self.sensor_colorscale,
                    cmin=self.field_cmin,
                    cmax=self.field_cmax,
                    flatshading=True,
                    lighting=dict(ambient=0.45, diffuse=0.75, specular=0.6),
                    lightposition=dict(x=120, y=160, z=180),
                    name=f"传感器 S{idx + 1}",
                    hoverinfo='text',
                    hovertext=[f"S{idx + 1} | 等待数据"] * len(vertices),
                    opacity=0.98,
                    showscale=False,
                )
                fig.add_trace(mesh_trace)
                self._sensor_mesh_indices.append(len(fig.data) - 1)
            if include_sensor_labels and label_positions:
                # 为传感器创建彩色 marker + 文本 trace，使四个传感器易于区分
                # 使用简单安全的 Scatter3d traces（不使用 UI 元素），避免在绘图函数中创建界面组件。
                default_colors = getattr(self, 'sensor_identity_colors', None) or ['#ef4444', '#f59e0b', '#10b981', '#3b82f6']
                xs = [float(p[0]) for p in label_positions]
                ys = [float(p[1]) for p in label_positions]
                zs = [float(p[2]) for p in label_positions]
                try:
                    fig.add_trace(go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode='markers+text',
                        marker=dict(size=6, color=default_colors[:len(xs)], opacity=1),
                        text=[f"S{i+1}" for i in range(len(xs))],
                        textposition='top center',
                        name='传感器标签',
                        hoverinfo='text'
                    ))
                except Exception:
                    try:
                        self.add_output('添加传感器标签失败', level='WARNING')
                    except Exception:
                        pass
        

    def _compute_sensor_grid_indices(self) -> tuple[int | None, int | None, int | None, int | None]:
        """根据传感器坐标估计 2x2 布局，返回 (tl, tr, bl, br) 索引。失败时返回 (None, None, None, None)。"""
        try:
            sp = np.asarray(self.sensor_positions, dtype=float)
            if sp.shape[0] < 4:
                return None, None, None, None
            by_y_desc = np.argsort(-sp[:, 1])
            top2 = by_y_desc[:2]
            bot2 = by_y_desc[2:4]
            tl_tr = top2[np.argsort(sp[top2, 0])]
            bl_br = bot2[np.argsort(sp[bot2, 0])]
            tl, tr = int(tl_tr[0]), int(tl_tr[1])
            bl, br = int(bl_br[0]), int(bl_br[1])
            return tl, tr, bl, br
        except Exception:
            return None, None, None, None

    def _update_magnets_for_plot(self, plot_component) -> None:
        """将当前 self.magnet_positions 同步到给定的 Plotly 面板组件。"""
        try:
            if not plot_component or not hasattr(plot_component, 'figure') or plot_component.figure is None:
                return
            fig = plot_component.figure
            magnet_idx = None
            for i, tr in enumerate(fig.data):
                try:
                    if getattr(tr, 'name', '') == '磁场源':
                        magnet_idx = i
                        break
                except Exception:
                    continue

            mx, my, mz = [], [], []
            if getattr(self, 'magnet_positions', None) is not None and self.magnet_positions.size > 0:
                mp = np.asarray(self.magnet_positions, dtype=float)
                mx = mp[:, 0].tolist(); my = mp[:, 1].tolist(); mz = mp[:, 2].tolist()

            if magnet_idx is not None and 0 <= magnet_idx < len(fig.data):
                tr = fig.data[magnet_idx]
                try:
                    tr.x = mx; tr.y = my; tr.z = mz
                    tr.visible = True if mx else False
                except Exception:
                    pass
            else:
                if mx:
                    try:
                        fig.add_trace(go.Scatter3d(
                            x=mx, y=my, z=mz,
                            mode='markers',
                            marker=dict(size=8, color='#10b981', symbol='diamond', line=dict(color='#fde68a', width=1.3)),
                            name='磁场源',
                            hoverinfo='text',
                            hovertext=[f"F{i + 1} | 磁场源" for i in range(len(mx))]
                        ))
                    except Exception:
                        pass

            try:
                plot_component.update()
            except Exception:
                pass
        except Exception:
            pass

    def _draw_streamlines_for_mode(self, mode: str = 'global') -> None:
        """该可视化函数已移除：此函数为占位，保持接口兼容。"""
        return

    def _remove_streamline_traces(self) -> None:
        """该可视化函数已移除：此函数为占位，不执行任何操作。"""
        return

    def scan_i2c_bus(self, start_addr: int = 0x03, end_addr: int = 0x77, timeout: float = 0.06) -> list:
        """扫描 I2C 总线，尝试检测 MLX90393 设备的地址。

        返回检测到的地址列表（整数，升序按检测顺序）。实现为对每个地址尝试一次快速测量读取，
        能够快速过滤不存在的从机。该函数会捕获所有通信错误并继续扫描。
        """
        found = []
        try:
            # 限制扫描范围并跳过保留地址
            start = max(0x03, int(start_addr))
            end = min(0x77, int(end_addr))
            for a in range(start, end + 1):
                try:
                    mlx = MLX90393(self.ch347, a, debug=False)
                    # 尝试快速启动测量并读取一次（短超时）
                    try:
                        mlx.start_poll_measurement(retries=1)
                        time.sleep(timeout)
                        _ = mlx.read_measurement(retries=1)
                        # 如果没有抛出异常，则认为设备存在且响应为 MLX90393
                        found.append(a)
                    except Exception:
                        # 非 MLX90393 或不响应 - 忽略
                        pass
                except Exception:
                    # 构造或 I2C 设置失败时忽略该地址
                    continue
        except Exception:
            pass
        return found




    def _safe_update_plot(self, new_fig):
        """安全更新图表的辅助方法，减少ResizeObserver错误"""
        try:
            if hasattr(self, 'three_d_plot') and self.three_d_plot:
                # 避免重复设置样式，减少尺寸计算触发
                try:
                    # 只更新figure内容，不重复设置样式
                    self.three_d_plot.figure = new_fig
                except:
                    # 如果直接更新失败，尝试重新创建图表组件
                    logger.warning("直接更新失败，尝试重新创建图表")
            
            if hasattr(self, 'measure_status') and self.measure_status:
                try:
                    self.measure_status.text = "测量中"
                    self.measure_status.classes("text-success font-medium text-sm")
                except:
                    pass
        except Exception as e:
            logger.warning(f"安全更新失败: {str(e)}")
    
    # 相关可视化项已移除：保留占位方法以避免其他调用导致错误（返回空结果）
    def generate_dynamic_field_lines(self, latest_data: dict[str, dict[str, float]], phase: float = 0.0):
        return [], {}

    def update_3d_magnetic_field_data(self, frame_data: dict[str, dict[str, float]] | None = None, advance_phase: bool = False) -> None:
        """安全占位：更新三维磁场数据的接口。
        如果类中已有更完整实现会被替换；当前实现尽量安静地处理并避免抛出异常或记录 WARNING。
        """
        try:
            # 如果提供了帧数据且存在绘图更新函数，尝试使用它
            if frame_data is not None:
                try:
                    # 若有 create_3d_magnetic_field_plot/_safe_update_plot 等方法，尝试刷新图表
                    new_fig = None
                    try:
                        new_fig = self.create_3d_magnetic_field_plot(include_sensor_labels=False, frame_data=frame_data)
                    except Exception:
                        try:
                            new_fig = self.create_3d_magnetic_field_plot(include_sensor_labels=False)
                        except Exception:
                            new_fig = None

                    if new_fig is not None and hasattr(self, '_safe_update_plot'):
                        try:
                            self._safe_update_plot(new_fig)
                        except Exception:
                            pass
                except Exception:
                    # 不要记录警告，保持静默
                    pass

            # 若没有 frame_data，仅尝试调用已有的刷新逻辑（若存在），但保持静默
            else:
                try:
                    if hasattr(self, '_safe_update_plot'):
                        # 试图使用当前 three_d_plot.figure 刷新（如有需要，子类可覆盖）
                        try:
                            cur_fig = getattr(self, 'three_d_plot').figure if getattr(self, 'three_d_plot', None) else None
                        except Exception:
                            cur_fig = None
                        if cur_fig is not None:
                            try:
                                self._safe_update_plot(cur_fig)
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception:
            # 最终兜底：不抛出、不记录WARNING
            return

    def build_ui(self) -> None:
        """构建完整的NiceGUI界面"""
        # 注入深色主题CSS样式
        ui.add_head_html('''
        <style>
            :root {
                --q-primary: #0f172a;
                --q-secondary: #1e293b;
                --q-tertiary: #334155;
                --q-accent: #06b6d4;
                --q-positive: #22c55e;
                --q-negative: #ef4444;
                --q-warning: #f59e0b;
                --q-info: #3b82f6;
                --q-light: #f8fafc;
                --q-dark: #0f172a;
            }
            
            body {
                background-color: #0f172a;
                color: #e2e8f0;
            }
            
            .bg-primary { background-color: #0f172a !important; }
            .bg-secondary { background-color: #1e293b !important; }
            .bg-tertiary { background-color: #334155 !important; }
            
            .text-primary { color: #e2e8f0 !important; }
            .text-secondary { color: #cbd5e1 !important; }
            .text-muted { color: #94a3b8 !important; }
            
            .border-primary { border-color: #334155 !important; }
            .border-secondary { border-color: #475569 !important; }
            
            .btn-primary { background-color: #3b82f6 !important; color: #fff !important; border: none !important; }
            .btn-secondary { background-color: #475569 !important; color: #e2e8f0 !important; border: none !important; }
            .btn-success { background-color: #22c55e !important; color: #000 !important; border: none !important; }
            .btn-warning { background-color: #f59e0b !important; color: #000 !important; border: none !important; }
            .btn-danger { background-color: #ef4444 !important; color: #fff !important; border: none !important; }
            .btn-ghost { background-color: transparent !important; color: #e2e8f0 !important; border: 1px solid #334155 !important; }
            
            .text-success { color: #22c55e !important; }
            .text-error { color: #ef4444 !important; }
            
            .compact-label { line-height: 1.2; }
            .compact-btn { min-height: 32px; }
            
            .control-area { box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); }
            
            .full-panel { position: fixed; top: 0; left: 0; right: 0; bottom: 0; z-index: 999; width: 100vw; height: 100vh; margin: 0; padding: 0; overflow: auto; }
            .panel-hidden { display: none !important; }
        </style>
        ''')
        
        # 1. 页面标题 - 深色主题学术风格
        with ui.header(elevated=False).classes("bg-primary border-b border-primary"):
            with ui.row().classes("items-center justify-center w-full py-8"):
                    with ui.column().classes("items-center gap-3"):
                        ui.label("磁场传感器测量系统").classes("text-5xl font-semibold text-primary")
                        ui.label("Magnetic Field Sensor Measurement System").classes("text-lg text-secondary")

                        # 顶部导航按钮（已移除）

        # 在页面加载后注入全局 JS 函数用于切换面板全屏（减少内联 JS 冲突）
        # 延迟注入全局 JS，避免在应用尚未启动事件循环时调用 ui.run_javascript
        ui.timer(0.3, lambda: ui.run_javascript('''
        window.togglePanelFull = function(panelId){
            try{
                const panels = ['param_panel', 'data_panel', 'three_d_panel'];
                const tp = document.getElementById(panelId);
                if(!tp){ console.warn('togglePanelFull: not found', panelId); return; }
                const isFull = tp.classList.contains('full-panel');

                function exitFull(){
                    panels.forEach(function(id){ const el = document.getElementById(id); if(!el) return; el.classList.remove('full-panel'); el.classList.remove('panel-hidden'); });
                    const btn = document.getElementById('panel_close_btn'); if(btn) btn.remove();
                    if(window._panelEscHandler){ try{ window.removeEventListener('keydown', window._panelEscHandler); delete window._panelEscHandler; }catch(e){} }
                    // restore/show the original '放大' button (if present) and clear inline styles set when entering full
                        try{ var enterBtnR = document.getElementById('data_panel_full_btn'); if(!enterBtnR) enterBtnR = document.querySelector('[aria-label="全屏"]'); if(enterBtnR){ enterBtnR.style.display = ''; enterBtnR.style.visibility = ''; enterBtnR.style.pointerEvents = ''; } }catch(e){}
                    try{
                        panels.forEach(function(id){
                            const el = document.getElementById(id);
                            if(!el) return;
                            el.style.position = '';
                            el.style.top = '';
                            el.style.height = '';
                            el.style.left = '';
                            el.style.right = '';
                            el.style.overflow = '';
                            el.style.display = '';
                            el.style.padding = '';
                            el.style.flexDirection = '';
                            // clear card-level inline styles we set when entering full
                            try{
                                const cards = el.querySelectorAll('.card');
                                cards.forEach(function(c){ try{ c.style.height = ''; c.style.minHeight = ''; c.style.display = ''; c.style.flexDirection = ''; c.style.overflow = ''; c.style.padding = ''; }catch(e){} });
                            }catch(e){}
                            // clear plot container heights
                            try{
                                const plotContainers = el.querySelectorAll('.plotly-graph-div, [data-echarts-instance], .echarts');
                                plotContainers.forEach(function(pc){ try{ pc.style.height = ''; pc.style.minHeight = ''; pc.style.display = ''; }catch(e){} });
                            }catch(e){}
                        });
                        document.body.style.overflow = '';
                    }catch(e){}
                    setTimeout(function(){ try{ window.dispatchEvent(new Event('resize')); }catch(e){} }, 120);
                }

                if(!isFull){
                    panels.forEach(function(id){ const el = document.getElementById(id); if(!el) return; if(id === panelId){ el.classList.add('full-panel'); el.classList.remove('panel-hidden'); } else { el.classList.add('panel-hidden'); el.classList.remove('full-panel'); }});
                    document.body.style.overflow = 'hidden';
                    // allow scrolling inside the panel (vertical/horizontal) so content can be examined
                    try{ tp.style.overflow = 'auto'; tp.style.overflowX = 'auto'; tp.style.overflowY = 'auto'; }catch(e){}
                    window.scrollTo(0,0);

                    // compute header/footer heights and set panel top/height so it won't be hidden under header/footer
                    try{
                        const headerEl = document.querySelector('header');
                        var headerH = headerEl ? headerEl.getBoundingClientRect().height : 0;
                        const footerEl = document.querySelector('footer');
                        var footerH = footerEl ? footerEl.getBoundingClientRect().height : 0;
                        // apply inline styles to position below header and above footer
                        tp.style.position = 'fixed';
                        tp.style.top = headerH + 'px';
                        tp.style.left = '0';
                        tp.style.right = '0';
                        tp.style.height = (window.innerHeight - headerH - footerH) + 'px';
                        tp.style.overflow = 'auto';
                        tp.style.display = 'flex';
                        tp.style.flexDirection = 'column';
                        tp.style.padding = '8px';
                    }catch(e){}

                    // set reasonable heights for inner cards/plots
                    try{
                        var headerEl2 = document.querySelector('header');
                        var headerH2 = headerEl2 ? headerEl2.getBoundingClientRect().height : 80;
                        var availH = window.innerHeight - headerH2 - 40; // padding estimate
                        var perPlotH = Math.max(180, Math.floor((availH - 40) / 2)); // two rows
                        var cards = tp.querySelectorAll('.card');
                        cards.forEach(function(c){ try{ c.style.height = perPlotH + 'px'; c.style.minHeight = '120px'; c.style.display = 'flex'; c.style.flexDirection = 'column'; c.style.overflow = 'hidden'; }catch(e){} });
                        var plotContainers = tp.querySelectorAll('.plotly-graph-div, [data-echarts-instance], .echarts');
                        plotContainers.forEach(function(pc){ try{ pc.style.height = (perPlotH - 40) + 'px'; }catch(e){} });
                        // trigger multiple resizes to handle async rendering
                        setTimeout(function(){ try{ window.dispatchEvent(new Event('resize')); }catch(e){} }, 150);
                        setTimeout(function(){ try{ window.dispatchEvent(new Event('resize')); }catch(e){} }, 450);
                        setTimeout(function(){ try{ window.dispatchEvent(new Event('resize')); }catch(e){} }, 900);
                    }catch(e){}

                    if(!document.getElementById('panel_close_btn')){
                        const btn = document.createElement('button');
                        btn.id = 'panel_close_btn'; btn.innerText = '退出全屏';
                        btn.className = 'btn-secondary';
                        btn.style.position = 'fixed'; btn.style.right = '18px';
                        try{ var headerEl3 = document.querySelector('header'); var headerH3 = headerEl3 ? headerEl3.getBoundingClientRect().height : 18; btn.style.top = (headerH3 + 6) + 'px'; }catch(e){ btn.style.top = '18px'; }
                        btn.style.zIndex = '99999';
                        btn.style.background = '#ef4444'; btn.style.color = '#fff'; btn.style.border = 'none'; btn.style.padding = '8px 12px'; btn.style.borderRadius = '6px'; btn.style.cursor = 'pointer';
                        btn.onclick = exitFull;
                        document.body.appendChild(btn);
                        // hide the original '放大' button while in full screen (robust lookup)
                        try{ var enterBtn = document.getElementById('data_panel_full_btn'); if(!enterBtn) enterBtn = document.querySelector('[aria-label="全屏"]'); if(enterBtn){ enterBtn.style.display = 'none'; enterBtn.style.visibility = 'hidden'; enterBtn.style.pointerEvents = 'none'; } }catch(e){}
                    }

                    window._panelEscHandler = function(ev){ if(ev.key === 'Escape') exitFull(); };
                    window.addEventListener('keydown', window._panelEscHandler);

                    // final resize adjustments for Plotly/ECharts
                    setTimeout(function(){ try{
                        var headerHf = document.querySelector('header') ? document.querySelector('header').offsetHeight : 80;
                        var avail = window.innerHeight - headerHf - 120; // padding/controls estimate
                        var perPlotHf = Math.max(220, Math.floor(avail / 2) - 20);
                        const plotlyEls = tp.querySelectorAll('.plotly-graph-div');
                        plotlyEls.forEach(function(el){ try{ el.style.height = perPlotHf + 'px'; if(window.Plotly && window.Plotly.Plots && window.Plotly.Plots.resize) window.Plotly.Plots.resize(el); }catch(e){} });
                        const echartsEls = tp.querySelectorAll('[data-echarts-instance], .echarts');
                        echartsEls.forEach(function(el){ try{ el.style.height = perPlotHf + 'px'; if(window.echarts){ var inst = window.echarts.getInstanceByDom(el); if(inst) inst.resize(); } }catch(e){} });
                        window.dispatchEvent(new Event('resize'));
                    }catch(e){} }, 200);
                } else {
                    // restore any inline heights we set when entering full
                    try{
                        const tpEls = document.getElementById(panelId);
                        if(tpEls){
                            tpEls.style.overflow = '';
                            tpEls.style.overflowX = '';
                            tpEls.style.overflowY = '';
                            tpEls.style.top = '';
                            tpEls.style.height = '';
                            tpEls.style.left = '';
                            tpEls.style.right = '';
                            const plotlyEls = tpEls.querySelectorAll('.plotly-graph-div'); plotlyEls.forEach(function(el){ try{ el.style.height = ''; if(window.Plotly && window.Plotly.Plots && window.Plotly.Plots.resize) window.Plotly.Plots.resize(el); }catch(e){} });
                            const echartsEls = tpEls.querySelectorAll('[data-echarts-instance], .echarts'); echartsEls.forEach(function(el){ try{ el.style.height = ''; if(window.echarts){ var inst = window.echarts.getInstanceByDom(el); if(inst) inst.resize(); } }catch(e){} });
                        }
                    }catch(e){}
                    exitFull();
                }
            }catch(e){ console.error('togglePanelFull error', e); }
        };
        '''))

        # 注入高亮/滚动交互函数：当用户点击 PCB 示意的 S1..S4 时，触发该方法高亮对应图卡并滚动到视图
        ui.timer(0.4, lambda: ui.run_javascript('''
        window.highlightSensor = function(n){
            try{
                var el = document.getElementById('sensor_card_' + n);
                if(!el) return;
                // 平滑滚动到对应卡片
                try{ el.scrollIntoView({behavior: 'smooth', block: 'center'}); }catch(e){}
                // 临时高亮边框
                var prevOutline = el.style.outline;
                el.style.outline = '3px solid #f59e0b';
                el.style.transition = 'outline 0.2s ease';
                // 显示 PCB 面板内的地址信息（如存在）
                try{
                    var infoEl = document.getElementById('pcb-sensor-info');
                    var addrEl = document.getElementById('sensor_addr_' + n);
                    if(infoEl){
                        infoEl.innerText = addrEl ? addrEl.innerText : ('S' + n);
                    }
                }catch(e){}
                setTimeout(function(){ try{ el.style.outline = prevOutline || ''; if(document.getElementById('pcb-sensor-info')) document.getElementById('pcb-sensor-info').innerText=''; }catch(e){} }, 2200);
            }catch(e){ console.error(e); }
        };
        '''), once=True)

        # 2. 主内容区（深色主题布局）- 4列3行网格布局（已删除右侧三维面板）
        with ui.grid(columns=4, rows=3).classes("p-6 w-full gap-6 bg-primary").style(
            "gap: 1.5rem 1rem; grid-template-columns: 280px 1fr 1fr 1fr;"):
            # -------------------------- 左侧控制区（第1列，第1-2行）--------------------------
            with ui.card().props('id=param_panel').classes("col-span-1 row-span-2 bg-secondary border border-primary control-area"):
                with ui.row().classes("p-4 mb-4 border-b border-primary"):
                    ui.label("实验参数设置").classes("text-lg font-semibold text-primary compact-label")

                # 配置参数 - 深色主题设计（紧凑布局）
                with ui.column().classes("px-4 mb-6"):
                    ui.label("测量参数").classes("text-base font-medium text-primary mb-3 compact-label")

                    with ui.column().classes("gap-3"):
                        with ui.row().classes("items-center gap-3"):
                            ui.label("采样频率 (Hz)").classes("text-sm text-secondary w-20 font-medium compact-label")
                            # 频率默认从 config 中读取（向后兼容 sample_interval）
                            default_interval = float(self.config.get("sample_interval", 0.05)) if isinstance(self.config.get("sample_interval", None), (int, float)) else 0.05
                            default_freq = float(self.config.get("sample_frequency", 1.0 / default_interval if default_interval > 0 else 20.0))
                            self.sample_frequency = ui.number(
                                value=default_freq,
                                step=0.1, min=0.1
                            ).classes("flex-1 bg-tertiary border-primary compact-btn").props("id=sample_frequency_input")
                            # 最大采样频率显示与测试按钮
                            self.max_freq_label = ui.label("(未测试)").classes("text-xs text-secondary ml-2")
                            ui.button("严格测试并固定最大采样频率", on_click=lambda: threading.Thread(target=self.measure_max_sampling_frequency_rigorous, daemon=True).start()).classes("btn-primary ml-2")
                            # 当用户输入的频率超过已知最大值时提示
                            def _on_freq_change(e: events.ValueChangeEventArguments | None = None):
                                try:
                                    val = float(self.sample_frequency.value)
                                except Exception:
                                    return
                                maxf = getattr(self, 'max_sample_frequency', None)
                                if maxf and val > maxf:
                                    ui.notify('已超出系统最大采样频率，请重新设置', type='warning')
                                    # 不自动回退值，也不创建任何额外控件；仅提示用户手动调整
                            self.sample_frequency.on('change', _on_freq_change)

                        # 确认对话框将在页面底部以模态方式创建（在页面主布局外创建以确保可见性）

                        with ui.row().classes("items-center gap-3"):
                            ui.label("测量时长 (s)").classes("text-sm text-secondary w-20 font-medium compact-label")
                            self.measurement_duration = ui.number(
                                value=self.config.get("measurement_duration", 20),
                                step=1, min=1
                            ).classes("flex-1 bg-tertiary border-primary compact-btn")

                        with ui.row().classes("items-center gap-3"):
                            ui.label("日志级别").classes("text-sm text-secondary w-20 font-medium compact-label")
                            self.log_level = ui.select(
                                ["DEBUG", "INFO", "WARNING", "ERROR"],
                                value=self.config.get("log_level", "DEBUG"),
                                on_change=self.update_log_level
                            ).classes("flex-1 bg-tertiary border-primary compact-btn")

                # 传感器控制按钮（紧凑布局）
                with ui.column().classes("px-4 mb-6"):
                    ui.label("传感器控制").classes("text-base font-medium text-primary mb-3")
                    with ui.row().classes("gap-3"):
                        self.calibrate_btn = ui.button(
                            "校准",
                            on_click=self.calibrate_stray_field
                        ).classes("btn-secondary px-3 py-2 font-medium")

                        # 旋转采样校准按钮（在后台线程运行）
                        ui.button(
                            "旋转校准",
                            on_click=lambda: threading.Thread(target=lambda: self.calibrate_stray_field_rotation(samples=36, interval=0.2), daemon=True).start()
                        ).classes("btn-secondary px-3 py-2 font-medium")

                        self.save_config_btn = ui.button(
                            "保存",
                            on_click=self.save_config
                        ).classes("btn-secondary px-3 py-2 font-medium")

                # 测量控制按钮（紧凑布局）
                with ui.column().classes("px-4 mb-6"):
                    ui.label("测量控制").classes("text-base font-medium text-primary mb-3")
                    with ui.column().classes("gap-2"):
                        self.start_measure_btn = ui.button(
                            "开始测量",
                            on_click=self.start_measurement
                        ).classes("btn-primary font-medium px-4 py-2 w-full")

                        # 暂停和继续按钮行
                        with ui.row().classes("gap-2 w-full"):
                            self.pause_measure_btn = ui.button(
                                "暂停",
                                on_click=self.pause_measurement
                            ).classes("btn-warning font-medium px-3 py-2 flex-1")
                            self.pause_measure_btn.disable()

                            self.resume_measure_btn = ui.button(
                                "继续",
                                on_click=self.resume_measurement
                            ).classes("btn-success font-medium px-3 py-2 flex-1")
                            self.resume_measure_btn.disable()

                        self.stop_measure_btn = ui.button(
                            "停止测量",
                            on_click=self.stop_measurement
                        ).classes("btn-danger font-medium px-4 py-2 w-full")
                        self.stop_measure_btn.disable()

                # 进度条与状态（紧凑布局）
                with ui.column().classes("px-4 mb-6"):
                    ui.label("测量状态").classes("text-base font-medium text-primary mb-3")
                    self.status_label = ui.label("准备就绪").classes("text-sm mb-3 text-secondary")
                    self.progress_bar = ui.linear_progress(value=0.0).classes("w-full mb-4")

                # 数据操作按钮（紧凑布局）
                with ui.column().classes("px-4 gap-2"):
                    ui.button(
                        "保存数据",
                        on_click=self.save_measurement_data
                    ).classes("btn-secondary font-medium w-full py-2")
                    ui.button(
                        "清空输出",
                        on_click=self.clear_output
                    ).classes("btn-secondary font-medium w-full py-2")

                    # 显式把流线与自动拟合控件放到左侧控制区，确保在任何布局下都可见
                    # 左侧“三维控制”分组：已移除

            # -------------------------- 中间数据展示区（第2-4列，跨2行）--------------------------
            with ui.card().props('id=data_panel').classes("col-span-3 row-span-2 bg-secondary border border-primary"):
                with ui.row().classes("p-3 mb-2 border-b border-primary items-center"):
                    ui.label("数据展示区").classes("text-lg font-semibold text-primary")
                    # 放大/全屏按钮：单一入口，进入全屏时会隐藏，退出时恢复
                    ui.button("全屏", on_click=lambda: ui.run_javascript("togglePanelFull('data_panel')")).classes("btn-secondary ml-auto").props('id=data_panel_full_btn aria-label=全屏')

                    # 扫描 I2C 的后台处理与轮询（移植自 measure6.py）
                    def _on_scan_click(e=None):
                        # 启动后台扫描任务，只在后台线程执行 I2C 交互，不直接操作 UI。
                        if getattr(self, '_scan_in_progress', False):
                            ui.notify('已有扫描在进行中，请等待', type='warning')
                            return
                        try:
                            self._scan_in_progress = True
                            self._scan_result = None
                            try:
                                self.add_output('开始扫描 I2C 总线以识别 MLX90393 设备...')
                            except Exception:
                                pass
                            try:
                                ui.notify('正在扫描 I2C（请稍候）', type='info')
                            except Exception:
                                pass
                        except Exception:
                            pass

                        def _scan_worker():
                            try:
                                found = self.scan_i2c_bus()
                                self._scan_result = found
                            except Exception as ex:
                                self._scan_result = ex
                            finally:
                                self._scan_in_progress = False

                        threading.Thread(target=_scan_worker, daemon=True).start()

                        def _poll_scan():
                            try:
                                if getattr(self, '_scan_in_progress', False):
                                    return
                                result = getattr(self, '_scan_result', None)
                                if result is None:
                                    return

                                if isinstance(result, Exception):
                                    try:
                                        self.add_output(f'I2C 扫描失败: {result}', level='ERROR')
                                        ui.notify('I2C 扫描失败', type='error')
                                    except Exception:
                                        pass
                                else:
                                    found = result
                                    if not found:
                                        try:
                                            self.add_output('未发现 MLX90393 设备', level='WARNING')
                                            ui.notify('未发现 MLX90393 设备', type='warning')
                                        except Exception:
                                            pass
                                    else:
                                        try:
                                            cfg = {}
                                            if os.path.exists('config.json'):
                                                with open('config.json', 'r', encoding='utf-8') as f:
                                                    cfg = json.load(f)
                                        except Exception:
                                            cfg = {}
                                        cfg['sensor_address_sequence'] = [hex(a) for a in found]
                                        try:
                                            with open('config.json', 'w', encoding='utf-8') as f:
                                                json.dump(cfg, f, ensure_ascii=False, indent=4)
                                            self.add_output(f"扫描完成，已在 config.json 中写入 sensor_address_sequence: {[hex(a) for a in found]}")
                                        except Exception as ex:
                                            self.add_output(f"写入 config.json 失败: {ex}", level='ERROR')

                                        try:
                                            self.sensor_addresses = list(found)
                                            self.sensors = [MLX90393(self.ch347, addr) for addr in self.sensor_addresses]
                                            self.sensor_hex_sequence = [hex(a) for a in self.sensor_addresses]
                                            with self.data_lock:
                                                self.measurement_data = {
                                                    hex(addr): {"times": [], "system_times": [], "Ts": [], "Xs": [], "Ys": [], "Zs": [], "totals": []}
                                                    for addr in self.sensor_addresses
                                                }
                                            ui.notify(f"检测到 {len(found)} 个设备: {[hex(a) for a in found]}", type='positive')
                                            self.add_output(f"I2C 扫描完成: {[hex(a) for a in found]}")
                                            try:
                                                new_fig = self.create_3d_magnetic_field_plot(include_sensor_labels=False)
                                                self._safe_update_plot(new_fig)
                                            except Exception:
                                                pass
                                        except Exception as ex:
                                            self.add_output(f"更新传感器映射失败: {ex}", level='ERROR')

                                try:
                                    if getattr(self, '_scan_timer', None):
                                        try:
                                            self._scan_timer.cancel()
                                        except Exception:
                                            pass
                                        self._scan_timer = None
                                except Exception:
                                    pass
                            except Exception:
                                pass

                        try:
                            self._scan_timer = ui.timer(0.5, lambda: _poll_scan(), once=False)
                        except Exception:
                            ui.timer(0.5, lambda: _poll_scan(), once=True)

                    # 显示/隐藏 PCB 示意（直接切换元素显示）
                    ui.button('显示 PCB 示意', on_click=lambda e=None: ui.run_javascript("(function(){var el=document.getElementById('pcb-panel'); if(!el) return; el.style.display = (el.style.display === 'none' || el.style.display === '') ? 'block' : 'none'; })();")).classes('btn-secondary compact-btn')
                    ui.button('扫描 I2C', on_click=_on_scan_click).classes('btn-primary compact-btn')

                # 新布局：上面 2x2 四个传感器图，下面横排控制模块
                with ui.column().classes("w-full gap-3"):
                    # 顶部四图 2x2
                    with ui.grid(columns=2).classes("w-full gap-3"):
                        self.sensor_plots = []
                        for idx in [0, 1, 2, 3]:
                            # 每个传感器图卡增加 id，便于从 PCB 示意中高亮/滚动
                            with ui.card().props(f'id=sensor_card_{idx+1}').classes("bg-primary border border-secondary w-full").style(
                                    "height: 310px; min-width: 260px;"):
                                plot = ui.echart({
                                    'backgroundColor': 'transparent',
                                    'title': {
                                        'text': f'传感器 {idx + 1}',
                                        'left': 'center',
                                        'top': '2%',
                                        'textStyle': {'color': '#ffffff', 'fontSize': 14}
                                    },
                                    'legend': {
                                        'show': True,
                                        'top': '10%',
                                        'left': 'center',
                                        'textStyle': {'color': '#ffffff', 'fontSize': 12},
                                        'itemGap': 12
                                    },
                                    'tooltip': {
                                        'show': True,
                                        'trigger': 'axis',
                                        'axisPointer': {
                                            'type': 'cross',
                                            'label': {'backgroundColor': '#6a7985'}
                                        },
                                        'textStyle': {'color': '#ffffff'},
                                        'backgroundColor': 'rgba(20,20,30,0.9)'
                                    },
                                    'grid': {'top': '30%', 'left': '10%', 'right': '6%', 'bottom': '22%',
                                             'containLabel': True},
                                    'xAxis': {
                                        'type': 'category',
                                        'data': [],
                                        'axisLabel': {'color': '#ffffff'},
                                        'axisLine': {'lineStyle': {'color': '#888'}}
                                    },
                                    'yAxis': {
                                        'type': 'value',
                                        'name': '总磁场(μT)',
                                        'nameLocation': 'middle',
                                        'nameGap': 40,
                                        'nameTextStyle': {'color': '#ffffff'},
                                        'axisLabel': {'color': '#ffffff'},
                                        'axisLine': {'lineStyle': {'color': '#888'}},
                                        'splitLine': {'lineStyle': {'color': '#444'}}
                                    },
                                    'series': [
                                        {
                                            'type': 'line',
                                            'data': [],
                                            'smooth': True,
                                            'connectNulls': True,
                                            'animation': False,
                                            'lineStyle': {'width': 2},
                                            'showSymbol': False
                                        }
                                    ]
                                }).classes("w-full").style("height: 290px; width: 100%;")
                                # 为每个图也注入一个隐藏的地址占位（某些环境下可能为空），以便 JS 读取显示
                                # 隐藏地址元素在页面中用于显示 PCB 中点击的地址信息
                                try:
                                    addr = self.sensor_addresses[idx] if idx < len(self.sensor_addresses) else None
                                    if addr is not None:
                                        ui.label(f"{hex(addr)}").props(f'id=sensor_addr_{idx+1}').style('display:none')
                                except Exception:
                                    pass
                                self.sensor_plots.append(plot)

                    # 底部控制模块 横排
                    with ui.grid(columns=2).classes("w-full gap-3"):
                        with ui.card().classes("p-3 bg-tertiary border border-primary"):
                            ui.label("磁场强度极值").classes("text-sm text-secondary mb-2 font-medium")
                            with ui.grid(columns=2).classes("gap-2"):
                                ui.label("最大值 (μT)").classes("text-xs text-secondary")
                                self.magnetic_max_display = ui.label("-").classes("text-sm font-bold text-primary")
                                ui.label("最小值 (μT)").classes("text-xs text-secondary")
                                self.magnetic_min_display = ui.label("-").classes("text-sm font-bold text-primary")
                                # 帧内平均（X/Y/Z/总场）显示已移除
                                # 每个传感器单独平均值（S1..S4）——恢复显示
                                for i, addr in enumerate(self.sensor_addresses):
                                    ui.label(f"S{i + 1} 平均 (μT)").classes("text-xs text-secondary")
                                    lbl = ui.label("-").classes("text-sm font-bold text-primary")
                                    # 保证 sensor_avg_displays 列表与传感器顺序一致
                                    self.sensor_avg_displays.append(lbl)

                        with ui.card().classes("p-3 bg-tertiary border border-primary"):
                            ui.label("数据类型选择").classes("text-sm font-medium mb-2 text-secondary")
                            with ui.column().classes("gap-2"):
                                # 使用复选框允许在单个传感器图上显示多条曲线（最多 5 条）
                                self.cb_x = ui.checkbox('X轴', value=False, on_change=self.on_compare_checkbox_change)
                                self.cb_y = ui.checkbox('Y轴', value=False, on_change=self.on_compare_checkbox_change)
                                self.cb_z = ui.checkbox('Z轴', value=False, on_change=self.on_compare_checkbox_change)
                                self.cb_temp = ui.checkbox('温度', value=False, on_change=self.on_compare_checkbox_change)
                                self.cb_total = ui.checkbox('总场', value=True, on_change=self.on_compare_checkbox_change)
                                with ui.row().classes("items-center gap-2 mt-2"):
                                    ref_input = ui.input(label="参考温度(°C)", placeholder="例如 25.0").classes("w-32")
                                    ui.button("按参考温度校准", on_click=lambda: threading.Thread(target=lambda: self.calibrate_temp_to_reference(float(ref_input.value) if ref_input.value else 25.0), daemon=True).start()).classes("btn-primary")
                                    ui.button("保存配置", on_click=lambda: threading.Thread(target=self.save_config, daemon=True).start()).classes("btn-secondary")

           

            # -------------------------- 底部第三行（自定义比例 1:2:2） --------------------------
            # 使用内嵌 grid 在整行内按比例 1:2:2 布局三个模块（响应结果 | 状态监控 | 输出窗口）
            with ui.row().classes("col-span-4").style("display: grid; grid-template-columns: 1fr 2fr 2fr; gap: 1rem;"):
                # 响应结果 (占比 1)
                with ui.card().classes("bg-secondary border border-primary").style("height: 260px; grid-column: 1 / 2;"):
                    with ui.row().classes("p-3 mb-3 border-b border-primary"):
                        ui.label("响应结果").classes("text-sm font-semibold text-primary")
                    # 模拟传感器响应状态
                    with ui.grid(columns=2).classes("mb-4 gap-2 px-3"):
                        for i, addr in enumerate(self.sensor_addresses):
                            with ui.card().classes("p-2 text-center bg-tertiary border border-primary"):
                                ui.label(f"传感器 {i + 1}").classes("text-xs font-medium text-secondary")
                                ui.label("已连接").classes("text-success text-xs font-bold")

                # 状态监控 (占比 2)
                with ui.card().classes("bg-secondary border border-primary").style("height: 260px; grid-column: 2 / 3;"):
                    with ui.row().classes("p-3 mb-3 border-b border-primary"):
                        ui.label("状态监控").classes("text-sm font-semibold text-primary")
                    with ui.column().classes("px-3"):
                        with ui.row().classes("mb-1"):
                            ui.label("测量功能:").classes("w-16 text-xs font-medium text-secondary")
                            self.measure_status = ui.label("未激活").classes("text-muted font-bold text-xs")
                        with ui.row().classes("mb-1"):
                            ui.label("数据存储:").classes("w-16 text-xs font-medium text-secondary")
                            ui.label("就绪").classes("text-success font-bold text-xs")
                        with ui.row().classes("mb-1"):
                            ui.label("设备连接:").classes("w-16 text-xs font-medium text-secondary")
                            ui.label("正常").classes("text-success font-bold text-xs")

                # 输出窗口 (占比 2)
                with ui.card().classes("bg-secondary border border-primary").style("height: 260px; grid-column: 3 / 4;"):
                    with ui.row().classes("p-3 mb-3 border-b border-primary"):
                        ui.label("输出窗口").classes("text-sm font-semibold text-primary")
                    self.output_text = ui.textarea(
                        value="程序开始运行...\n"
                    ).classes(
                        "w-full h-24 font-mono text-xs bg-primary border border-secondary text-success mx-3 resize-none").props(
                        f"readonly id={self.output_text_id}")

        # 3. 页脚
                # 隐藏的 PCB 示意面板（由顶部按钮切换显示）
                # 可拖拽的小弹窗（初始定位在右上，支持拖动并有阴影/毛玻璃效果）
                with ui.card().props('id=pcb-panel').classes('bg-tertiary border border-primary').style(
                    "position: fixed; right: 28px; top: 80px; width: 220px; height: 260px; z-index: 100020; display: none; overflow: hidden; border-radius: 10px; box-shadow: 0 14px 30px rgba(2,6,23,0.6); backdrop-filter: blur(6px);"):
                        # 使用内边距确保内部小卡被完整包裹
                        with ui.column().style('padding: 8px; cursor: default; height:100%; box-sizing:border-box;'):
                                ui.html('''
                                <div style="width:100%; height:100%; box-sizing:border-box; display:flex; flex-direction:column;">
                                    <div id="pcb-panel-header" style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px; cursor:move;">
                                        <div style="color:#e2e8f0; font-size:13px; font-weight:600;">PCB 示意图</div>
                                        <button id="pcb-panel-close" onclick="(function(){var el=document.getElementById('pcb-panel'); if(el) el.style.display='none';})();" style="background:transparent;border:none;color:#e2e8f0;font-weight:700;cursor:pointer;">✕</button>
                                    </div>
                                    <div style="font-size:12px;color:#94a3b8;margin-bottom:6px;">（示意图）</div>
                                    <div style="flex:1; display:flex; align-items:center; justify-content:center;">
                                        <div style="background:#0b1726;border-radius:8px;padding:10px;display:flex;flex-direction:column;align-items:center;justify-content:center;width:180px;height:140px;box-sizing:border-box;margin-top:6px;margin-bottom:8px;position:relative;">
                                            <div style="position:relative;width:100%;height:100%;">
                                                <!-- 更均匀分布的四个传感器点 -->
                                                <!-- S1: top-left -->
                                                <div class="pcb-dot" onclick="window.highlightSensor(1)" style="position:absolute;left:18%;top:16%;width:22px;height:22px;border-radius:50%;background:#34d399;box-shadow:0 0 0 8px rgba(52,211,153,0.14);"></div>
                                                <div style="position:absolute;left:12%;top:34%;color:#cbd5e1;font-size:11px;">S1</div>
                                                <!-- S2: top-right -->
                                                <div class="pcb-dot" onclick="window.highlightSensor(2)" style="position:absolute;left:72%;top:16%;width:22px;height:22px;border-radius:50%;background:#34d399;box-shadow:0 0 0 8px rgba(52,211,153,0.14);"></div>
                                                <div style="position:absolute;left:66%;top:34%;color:#cbd5e1;font-size:11px;">S2</div>
                                                <!-- S3: bottom-left -->
                                                <div class="pcb-dot" onclick="window.highlightSensor(3)" style="position:absolute;left:18%;top:66%;width:22px;height:22px;border-radius:50%;background:#34d399;box-shadow:0 0 0 8px rgba(52,211,153,0.14);"></div>
                                                <div style="position:absolute;left:12%;top:84%;color:#cbd5e1;font-size:11px;">S3</div>
                                                <!-- S4: bottom-right -->
                                                <div class="pcb-dot" onclick="window.highlightSensor(4)" style="position:absolute;left:72%;top:66%;width:22px;height:22px;border-radius:50%;background:#34d399;box-shadow:0 0 0 8px rgba(52,211,153,0.14);"></div>
                                                <div style="position:absolute;left:66%;top:84%;color:#cbd5e1;font-size:11px;">S4</div>
                                            </div>
                                            <div style="margin-top:6px;color:#cbd5e1;font-size:11px;">连接器（插侧）</div>
                                        </div>
                                        <!-- 把地址显示移到面板外侧（面板相对于视窗为 fixed，因此此处绝对定位到面板左下角） -->
                                        <div id="pcb-sensor-info" style="position:absolute;left:12px;bottom:12px;padding:4px 8px;border-radius:6px;border:2px solid #ef4444;min-width:44px;min-height:20px;color:#ef4444;font-size:12px;display:flex;align-items:center;justify-content:center;background:rgba(239,68,68,0.04);z-index:100025;">
                                        </div>
                                    </div>
                                </div>
                                ''').props('sanitize=False')
        # 将需要的脚本注入到 body（避免在 ui.html 中包含 <script> 标签）
        ui.add_body_html('''
        <style>
        /* PCB 点的交互样式 */
        .pcb-dot { cursor: pointer; transition: transform 0.12s ease, box-shadow 0.12s ease; }
        .pcb-dot:hover { transform: scale(1.12); box-shadow: 0 0 0 8px rgba(52,211,153,0.14); }
        /* 小面板微调，确保可拖动区域视觉友好 */
        #pcb-panel { position: fixed; z-index: 9999; }
        #pcb-panel-header { cursor: move; }
        </style>
        <script>
        (function(){
            function makeDraggable(headerId, panelId){
                var header = document.getElementById(headerId);
                var panel = document.getElementById(panelId);
                if(!header || !panel) return;
                var dragging = false, offsetX = 0, offsetY = 0;
                header.addEventListener('mousedown', function(e){
                    dragging = true;
                    var rect = panel.getBoundingClientRect();
                    offsetX = e.clientX - rect.left;
                    offsetY = e.clientY - rect.top;
                    document.body.style.userSelect = 'none';
                });
                document.addEventListener('mousemove', function(e){
                    if(!dragging) return;
                    var left = e.clientX - offsetX;
                    var top = e.clientY - offsetY;
                    left = Math.max(8, Math.min(window.innerWidth - panel.offsetWidth - 8, left));
                    top = Math.max(8, Math.min(window.innerHeight - panel.offsetHeight - 8, top));
                    panel.style.left = left + 'px';
                    panel.style.top = top + 'px';
                    panel.style.right = 'auto';
                    panel.style.position = 'fixed';
                });
                document.addEventListener('mouseup', function(){ if(dragging){ dragging = false; document.body.style.userSelect = ''; }});
            }

            function ensureHighlightSensor(){
                if(window.highlightSensor && typeof window.highlightSensor === 'function') return;
                window.highlightSensor = function(n){
                    try{
                        var el = document.getElementById('sensor_card_' + n);
                        if(!el) return;
                        try{ el.scrollIntoView({behavior: 'smooth', block: 'center'}); }catch(e){}
                        var prevOutline = el.style.outline;
                        el.style.outline = '3px solid #f59e0b';
                        el.style.transition = 'outline 0.2s ease';
                        try{
                            var infoEl = document.getElementById('pcb-sensor-info');
                            var addrEl = document.getElementById('sensor_addr_' + n);
                            if(infoEl){ infoEl.innerText = addrEl ? addrEl.innerText.toUpperCase() : ('S' + n); }
                        }catch(e){}
                        setTimeout(function(){ try{ el.style.outline = prevOutline || ''; if(document.getElementById('pcb-sensor-info')) document.getElementById('pcb-sensor-info').innerText=''; }catch(e){} }, 2200);
                    }catch(e){ console.error(e); }
                };
            }

            function attachCloseAndDraggable(){
                var closeBtn = document.getElementById('pcb-panel-close');
                if(closeBtn && !closeBtn._hasListener){
                    closeBtn.addEventListener('click', function(){ var el = document.getElementById('pcb-panel'); if(el) el.style.display = 'none'; var info = document.getElementById('pcb-sensor-info'); if(info) info.innerText = ''; });
                    closeBtn._hasListener = true;
                }
                makeDraggable('pcb-panel-header','pcb-panel');
            }

            if(document.readyState === 'loading'){
                document.addEventListener('DOMContentLoaded', function(){ ensureHighlightSensor(); attachCloseAndDraggable(); });
            } else {
                ensureHighlightSensor(); attachCloseAndDraggable();
            }
        })();
        </script>
        ''')

        # 顶层确认对话框（用于在严格测试后询问用户是否应用测试值）
        # 注意：某些 NiceGUI 版本不支持 modal 参数，故不传该参数以保证兼容性
        with ui.dialog() as self.apply_confirm_dialog:
            with ui.card().classes("p-4"):
                self.apply_dialog_label = ui.label("")
                with ui.row().classes("items-center gap-2 mt-4"):
                    ui.button("应用", on_click=lambda: self._apply_pending_max_freq()).classes("btn-primary")
                    ui.button("取消", on_click=lambda: self.apply_confirm_dialog.close()).classes("btn-secondary")

    def _start_detect_magnets(self) -> None:
        """兼容占位：原磁体检测功能已移除。"""
        self.add_output('磁体检测功能已移除', level='WARNING')

    def _poll_detect_results(self) -> None:
        return

    def _plot_detected_magnets(self, results: list) -> None:
        return

    def add_output(self, message: str, level: str = "INFO") -> None:
        """添加输出信息到日志窗口，并写入日志系统"""
        # 日志系统记录
        log_func = {
            "DEBUG": logger.debug,
            "INFO": logger.info,
            "WARNING": logger.warning,
            "ERROR": logger.error
        }.get(level, logger.info)
        log_func(message)
        # UI输出窗口显示（带时间戳）
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        new_content = f"[{timestamp}] [{level}] {message}\n"
        self.output_text.value += new_content

        # 尝试多种自动滚动方案
        try:
            # 方案1: 直接设置textarea的scrollTop
            ui.run_javascript(f"""
                setTimeout(function() {{
                    var textareas = document.querySelectorAll('textarea');
                    for (var i = 0; i < textareas.length; i++) {{
                        var textarea = textareas[i];
                        if (textarea.value && textarea.value.includes('[{level}]')) {{
                            textarea.scrollTop = textarea.scrollHeight;
                            break;
                        }}
                    }}
                }}, 50);
            """)
        except:
            pass


# -------------------------- 程序入口 --------------------------
def main():
    try:
        # 初始化CH347设备
        ch347 = CH347Device(usb_dev=0)
        if not ch347.openflag:
            logger.error("无法打开CH347设备，程序退出")
            ui.notify("无法打开CH347设备，请检查连接", type="error")
            sys.exit(1)
        ch347.get_dev_info()

        # 传感器地址列表（与measure5.py一致：0x0C, 0x0D, 0x0E, 0x0F）
        sensor_addresses = [0x0C, 0x0D, 0x0E, 0x0F]

        # 创建UI实例并构建界面
        app.title = "磁传感力学传感器测量系统"
        measurement_ui = MagneticMeasurementUI(ch347, sensor_addresses)
        measurement_ui.build_ui()

        # 应用关闭时清理设备
        app.on_shutdown(lambda: ch347.close())

        # 运行NiceGUI应用 - 使用不同端口避免冲突
        ui.run(
            port=8091,  # 更改端口避免冲突
            dark=True,  # 深色模式
            reload=False,
            show=True,  # 自动打开浏览器
            host='127.0.0.1'  # 明确指定本地主机
        )

    except Exception as e:
        logger.critical(f"程序启动失败: {str(e)}", exc_info=True)
        ui.notify(f"程序启动失败: {str(e)}", type="error")
        sys.exit(1)


if __name__ == "__main__":
    main()
