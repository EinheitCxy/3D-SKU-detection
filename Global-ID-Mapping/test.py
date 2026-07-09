import json

path = "/home/chenxingyu/3D_Recognization/Global-ID-Mapping/api_result.json"

# ✅ 1. 用 'r' 模式读取（文本模式，自动处理编码）
with open(path, 'r', encoding='utf-8') as f:
    m = json.load(f)

# ✅ 2. 用不同的变量名避免冲突
json_strings = m["global_skus"]

# ✅ 3. 在循环内处理每个对象
m=0
for json_str in json_strings:

    k = json.loads(json_str)
    with open(f"/home/chenxingyu/3D_Recognization/Global-ID-Mapping/test/{m}.json", "w", encoding="utf-8") as f:
        json.dump(k, f, ensure_ascii=False, indent=4)
        m+=1

    print(k)  # 打印每个解析后的字典

# 或者收集所有结果到列表
# results = [json.loads(s) for s in json_strings]
# print(results)
