import pandas as pd

def process_csv_column(file_path, column_name):
    """
    读取 CSV 文件，提取指定列的前两位字符，并替换原始值。

    参数:
    file_path (str): CSV 文件的路径。
    column_name (str): 需要处理的列的名称，例如 'KPS+神经功能 级（6月）'。
    """
    try:
        # 读取 CSV 文件，确保编码正确，避免中文乱码
        df = pd.read_csv(file_path, encoding='utf-8')

        # 检查指定的列是否存在
        if column_name not in df.columns:
            print(f"错误: CSV文件中未找到列 '{column_name}'。请检查列名是否正确。")
            return

        # 提取指定列的值，并转换为字符串类型
        # 然后使用 .str[:2] 提取每个值的前两位字符
        # .fillna('') 是为了处理空值（NaN），将其视为空字符串
        df[column_name] = df[column_name].astype(str).str[:2].fillna('')

        # 将处理后的 DataFrame 写入原始 CSV 文件
        # index=False 避免将索引写入文件
        df.to_csv(file_path, index=False, encoding='utf-8')

        print(f"成功处理文件 '{file_path}' 中的 '{column_name}' 列，并已保存。")

    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。请检查文件路径是否正确。")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

# 示例使用
if __name__ == "__main__":
    # 替换成你的 CSV 文件路径和需要处理的列名
    input_csv_path = "肺癌统计.csv"
    target_column = "KPS+神经功能 级（6月）"

    process_csv_column(input_csv_path, target_column)