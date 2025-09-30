import os
import json

input_dir = '.'  # 当前目录
output_root = './output'

# 遍历当前目录下所有 .json 文件
for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        filepath = os.path.join(input_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                output_dict = {}
                if "reference" in data:
                    for _, arxiv_id in data["reference"].items():
                        main_id = arxiv_id.split('v')[0]
                        output_dict[main_id] = {"arxivId": main_id}

                # 构造输出路径
                filename_no_ext = os.path.splitext(filename)[0]
                output_path = os.path.join(output_root, filename_no_ext, 'exp_1')
                os.makedirs(output_path, exist_ok=True)

                # 写入 ref.json
                output_file = os.path.join(output_path, 'ref.json')
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    json.dump(output_dict, out_f, indent=4, ensure_ascii=False)

                print(f"已生成：{output_file}")

        except Exception as e:
            print(f"处理文件 {filename} 出错：{e}")
