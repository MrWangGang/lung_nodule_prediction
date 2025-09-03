import pandas as pd

try:
    # --- 1. 读取两个 CSV 文件 ---
    # 读取 '肺癌统计.csv' 作为主数据框
    df_main = pd.read_csv('靶向_数据集.csv')

    # 读取 '输出文件.csv' 作为需要合并的数据
    df_to_merge = pd.read_csv('source.csv')

    # --- 2. 检查列名 ---
    # 假设 '输出文件.csv' 中需要合并的列名为 '3月肿瘤控制(未控制=1，控制=2)'
    column_to_add = '实体瘤评价(CR=1,PR=2,SD=3,PD=4)'
    if column_to_add not in df_to_merge.columns:
        print(f"错误：'输出文件.csv' 中未找到列名 '{column_to_add}'。请检查文件名是否正确。")
    else:
        # --- 3. 执行合并 ---
        # 直接将 '输出文件.csv' 的目标列拼接到 '肺癌统计.csv' 的最后一列
        # 这里假设两个文件的数据行数相同且顺序一致。
        # 如果顺序不一致，需要一个共同的标识符（如患者ID）进行精确匹配。

        # 获取目标列
        data_to_append = df_to_merge[column_to_add]

        # 将新列拼接到主数据框的最后一列
        df_main[column_to_add] = data_to_append

        # --- 4. 保存合并后的文件 ---
        # 你可以选择覆盖原文件或保存为新文件
        # 保存为新文件，以防万一
        output_filename = '肺癌统计.csv'
        df_main.to_csv(output_filename, index=False, encoding='utf-8-sig')

        print(f"成功合并数据。新文件已保存为 '{output_filename}'。")

except FileNotFoundError as e:
    print(f"错误：文件未找到。请确保文件 '{e.filename}' 位于正确路径。")
except Exception as e:
    print(f"发生了一个错误：{e}")