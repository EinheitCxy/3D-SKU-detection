uv #!/usr/bin/env python3
"""简单的连接测试脚本"""
import requests

API_URL = "http://localhost:8010"

print("测试 API 连接...")
print(f"目标地址: {API_URL}")

try:
    # 测试根路径
    response = requests.get(API_URL, timeout=5)
    print(f"✓ 根路径响应: {response.status_code}")

    # 测试 /api 路径（应该返回 405 Method Not Allowed，因为只支持 POST）
    response = requests.get(f"{API_URL}/api", timeout=5)
    print(f"✓ /api 路径响应: {response.status_code} (405 是正常的，因为只支持 POST)")

    print("\n✓ API 服务可以访问！")
    print("现在可以运行 test_api.py 进行完整测试")

except requests.exceptions.ConnectionError as e:
    print(f"✗ 连接失败: {e}")
    print("\n请检查:")
    print("1. Docker 容器是否运行: docker ps | grep global-id-mapping")
    print("2. 端口映射是否正确: 应该看到 0.0.0.0:8010->8010/tcp")
    print("3. 防火墙是否阻止了端口 8010")

except Exception as e:
    print(f"✗ 发生错误: {e}")
