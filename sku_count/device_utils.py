#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能设备选择工具
支持CUDA、MPS、CPU的自动选择和手动指定
"""

import torch
import logging
from typing import Optional, Union
import platform
import warnings

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeviceSelector:
    """智能设备选择器"""
    
    def __init__(self):
        self._device_cache = None
        self._device_info_cache = None
    
    def get_optimal_device(self, prefer_device: Optional[str] = None, 
                          force_cpu: bool = False, 
                          verbose: bool = True) -> torch.device:
        """
        获取最优设备
        
        Args:
            prefer_device: 偏好设备 ('cuda', 'mps', 'cpu' 或 None)
            force_cpu: 强制使用CPU
            verbose: 是否显示详细信息
            
        Returns:
            torch.device: 选择的设备
        """
        if force_cpu:
            device = torch.device('cpu')
            if verbose:
                logger.info("🖥️  强制使用CPU")
            return device
        
        # 如果有缓存且没有指定偏好，直接返回缓存
        if self._device_cache is not None and prefer_device is None:
            return self._device_cache
        
        # 设备选择优先级: prefer_device > CUDA > MPS > CPU
        if prefer_device:
            device = self._try_specific_device(prefer_device, verbose)
            if device:
                self._device_cache = device
                return device
        
        # 自动选择：CUDA > MPS > CPU
        device = self._auto_select_device(verbose)
        self._device_cache = device
        return device
    
    def _try_specific_device(self, device_name: str, verbose: bool = True) -> Optional[torch.device]:
        """尝试使用指定设备"""
        device_name = device_name.lower()
        
        if device_name == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                if verbose:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    logger.info(f"🚀 使用CUDA设备: {gpu_name} ({gpu_memory:.1f}GB)")
                return device
            else:
                if verbose:
                    logger.warning("⚠️  CUDA不可用，尝试其他设备")
                
        elif device_name == 'mps':
            if torch.backends.mps.is_available():
                device = torch.device('mps')
                if verbose:
                    logger.info("🍎 使用MPS设备 (Apple Silicon GPU)")
                return device
            else:
                if verbose:
                    logger.warning("⚠️  MPS不可用，尝试其他设备")
                    
        elif device_name == 'cpu':
            device = torch.device('cpu')
            if verbose:
                logger.info("🖥️  使用CPU设备")
            return device
        
        return None
    
    def _auto_select_device(self, verbose: bool = True) -> torch.device:
        """自动选择最优设备"""
        # 1. 尝试CUDA
        if torch.cuda.is_available():
            device = torch.device('cuda')
            if verbose:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"🚀 自动选择CUDA: {gpu_name} ({gpu_memory:.1f}GB)")
            return device
        
        # 2. 尝试MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            if verbose:
                system_info = platform.platform()
                logger.info(f"🍎 自动选择MPS: Apple Silicon GPU ({system_info})")
            return device
        
        # 3. 回退到CPU
        device = torch.device('cpu')
        if verbose:
            import psutil
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / 1024**3
            logger.info(f"🖥️  自动选择CPU: {cpu_count}核心, {memory_gb:.1f}GB内存")
        
        return device
    
    def get_device_info(self) -> dict:
        """获取设备详细信息"""
        if self._device_info_cache is not None:
            return self._device_info_cache
        
        info = {
            'cuda_available': torch.cuda.is_available(),
            'mps_available': torch.backends.mps.is_available(),
            'cpu_available': True,
            'devices': []
        }
        
        # CUDA信息
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info['devices'].append({
                    'type': 'cuda',
                    'id': i,
                    'name': props.name,
                    'memory_gb': props.total_memory / 1024**3,
                    'compute_capability': f"{props.major}.{props.minor}"
                })
        
        # MPS信息
        if torch.backends.mps.is_available():
            info['devices'].append({
                'type': 'mps',
                'id': 0,
                'name': 'Apple Silicon GPU',
                'platform': platform.platform()
            })
        
        # CPU信息
        import psutil
        info['devices'].append({
            'type': 'cpu',
            'id': 0,
            'name': platform.processor() or 'CPU',
            'cores': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / 1024**3
        })
        
        self._device_info_cache = info
        return info
    
    def print_device_info(self):
        """打印设备信息"""
        info = self.get_device_info()
        
        print("🔍 可用设备信息:")
        print("=" * 50)
        
        for device in info['devices']:
            if device['type'] == 'cuda':
                print(f"🚀 CUDA设备 {device['id']}: {device['name']}")
                print(f"   内存: {device['memory_gb']:.1f}GB")
                print(f"   计算能力: {device['compute_capability']}")
            elif device['type'] == 'mps':
                print(f"🍎 MPS设备: {device['name']}")
                print(f"   平台: {device['platform']}")
            elif device['type'] == 'cpu':
                print(f"🖥️  CPU设备: {device['name']}")
                print(f"   核心数: {device['cores']}")
                print(f"   内存: {device['memory_gb']:.1f}GB")
            print()
    
    def test_device_performance(self, device: torch.device, matrix_size: int = 1000) -> float:
        """测试设备性能"""
        try:
            import time
            
            # 创建测试矩阵
            a = torch.randn(matrix_size, matrix_size, device=device)
            b = torch.randn(matrix_size, matrix_size, device=device)
            
            # 预热
            for _ in range(3):
                _ = torch.mm(a, b)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()
            
            # 性能测试
            start_time = time.time()
            for _ in range(10):
                result = torch.mm(a, b)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()
            
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            return avg_time
            
        except Exception as e:
            logger.warning(f"设备性能测试失败 {device}: {e}")
            return float('inf')
    
    def benchmark_devices(self, matrix_size: int = 1000) -> dict:
        """基准测试所有可用设备"""
        print("🏃 开始设备性能基准测试...")
        print("=" * 50)
        
        results = {}
        
        # 测试CUDA
        if torch.cuda.is_available():
            device = torch.device('cuda')
            time_taken = self.test_device_performance(device, matrix_size)
            results['cuda'] = time_taken
            print(f"🚀 CUDA: {time_taken:.4f}秒")
        
        # 测试MPS
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            time_taken = self.test_device_performance(device, matrix_size)
            results['mps'] = time_taken
            print(f"🍎 MPS: {time_taken:.4f}秒")
        
        # 测试CPU
        device = torch.device('cpu')
        time_taken = self.test_device_performance(device, matrix_size)
        results['cpu'] = time_taken
        print(f"🖥️  CPU: {time_taken:.4f}秒")
        
        # 找出最快的设备
        if results:
            fastest = min(results.items(), key=lambda x: x[1])
            print(f"\n🏆 最快设备: {fastest[0].upper()} ({fastest[1]:.4f}秒)")
        
        return results
    
    def clear_cache(self):
        """清除缓存"""
        self._device_cache = None
        self._device_info_cache = None

# 全局设备选择器实例
_global_selector = DeviceSelector()

# 导出标志，供其他模块检查
DEVICE_UTILS_AVAILABLE = True

def get_optimal_device(prefer_device: Optional[str] = None, 
                      force_cpu: bool = False, 
                      verbose: bool = True) -> torch.device:
    """
    便捷函数：获取最优设备
    
    Args:
        prefer_device: 偏好设备 ('cuda', 'mps', 'cpu' 或 None)
        force_cpu: 强制使用CPU
        verbose: 是否显示详细信息
        
    Returns:
        torch.device: 选择的设备
        
    Examples:
        >>> device = get_optimal_device()  # 自动选择
        >>> device = get_optimal_device('mps')  # 偏好MPS
        >>> device = get_optimal_device(force_cpu=True)  # 强制CPU
    """
    return _global_selector.get_optimal_device(prefer_device, force_cpu, verbose)

def print_device_info():
    """便捷函数：打印设备信息"""
    _global_selector.print_device_info()

def benchmark_devices(matrix_size: int = 1000) -> dict:
    """便捷函数：基准测试设备"""
    return _global_selector.benchmark_devices(matrix_size)

def get_device_info() -> dict:
    """便捷函数：获取设备信息"""
    return _global_selector.get_device_info()

# 向后兼容的函数
def select_device(prefer_device: Optional[str] = None) -> torch.device:
    """向后兼容：选择设备"""
    warnings.warn("select_device已废弃，请使用get_optimal_device", DeprecationWarning)
    return get_optimal_device(prefer_device)

if __name__ == "__main__":
    # 演示用法
    print("🔧 PyTorch设备选择工具")
    print("=" * 50)
    
    # 显示设备信息
    print_device_info()
    
    # 自动选择设备
    print("自动选择最优设备:")
    device = get_optimal_device()
    print(f"选择的设备: {device}")
    print()
    
    # 基准测试
    benchmark_devices()
    
    print("\n✅ 设备选择工具测试完成！")