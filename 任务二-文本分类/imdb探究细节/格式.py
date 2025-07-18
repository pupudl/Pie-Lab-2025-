# 读取原始JSON文件（每行一个JSON对象）
with open('C:/Users/78636/Desktop/Sarcasm_Headlines_Dataset.json', 'r', encoding='utf-8') as f:
    # 读取所有行，去除空行
    lines = [line.strip() for line in f if line.strip()]

# 将每行的JSON对象用逗号连接，首尾加上方括号，形成JSON数组
json_array = '[' + ',\n'.join(lines) + ']'

# 写入处理后的文件
with open('C:/Users/78636/Desktop/output.json', 'w', encoding='utf-8') as f:
    f.write(json_array)