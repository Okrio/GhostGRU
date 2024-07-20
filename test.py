import os
import fitz  # PyMuPDF
import shutil
from pathlib import Path 

# 需要安装shutil 和pathlib 包
# 查找首页码的规则改成查找“American Chemical Society”的下一行，从给的pdf中可以看出“American Chemical Society”下一行即是首页码
def extract_page_numbers_from_content(pdf_path):
    try:
        # 打开PDF文件
        pdf_document = fitz.open(pdf_path)

        # 获取第一页
        first_page = pdf_document[0]

        # 提取文本内容
        page_text = first_page.get_text()

        # 搜索页码信息
        lines = page_text.split('\n')
        for idx, line in enumerate(lines):
            # 修改寻找规则
            if line == 'American Chemical Society':
                if idx+1 > len(lines):
                    return "Failed to extract page numbers from the content."
                page_num_cand = lines[idx+1]
                start_page = page_num_cand
                return f"{start_page}"


        return "Failed to extract page numbers from the content."

    except Exception as e:
        return f"An error occurred: {e}"





def rename_pdf_to_text(pdf_filename, new_text):
    try:
        # 检查文件是否存在
        if not os.path.exists(pdf_filename):
            print(f"Error: File '{pdf_filename}' does not exist.")
            return

        # 新文件名
        new_name = f"{new_text}.pdf"
        # 将原有文件拷贝一份并重新命名成new_text.pdf
        # shutil.copy(pdf_filename, new_name)

        # 重命名文件
        os.rename(pdf_filename, new_name)

        print(f"Renamed '{pdf_filename}' to '{new_name}'")

    except Exception as e:
        print(f"Error: {e}")

def find_pdf(file_path):
    for file_name in Path(file_path).rglob("*.pdf"):
        file_name_tmp = str(file_name)
        tar_dir = os.path.split(file_name_tmp)[0]
        new_file_name = extract_page_numbers_from_content(file_name_tmp)

        tar_file_name = os.path.join(tar_dir, new_file_name)
        if new_file_name != "Failed to extract page numbers from the content.":
            rename_pdf_to_text(file_name_tmp,tar_file_name)


# 示例用法:
if __name__ == "__main__":
    # 输入是 待转换的文件所在的文件夹目录 输出的文件同时存在在该文件夹中，并以首页码的方式进行命名
    file_path = '/Users/okrio/Downloads/test_pdf_pagenumber/'# 输入待转换的文件所在的文件夹目录
    find_pdf(file_path)
    print('succeed!!!')