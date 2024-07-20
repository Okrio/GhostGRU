import os
import fitz  # PyMuPDF
import shutil
from pathlib import Path 

# ��Ҫ��װshutil ��pathlib ��
# ������ҳ��Ĺ���ĳɲ��ҡ�American Chemical Society������һ�У��Ӹ���pdf�п��Կ�����American Chemical Society����һ�м�����ҳ��
def extract_page_numbers_from_content(pdf_path):
    try:
        # ��PDF�ļ�
        pdf_document = fitz.open(pdf_path)

        # ��ȡ��һҳ
        first_page = pdf_document[0]

        # ��ȡ�ı�����
        page_text = first_page.get_text()

        # ����ҳ����Ϣ
        lines = page_text.split('\n')
        for idx, line in enumerate(lines):
            # �޸�Ѱ�ҹ���
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
        # ����ļ��Ƿ����
        if not os.path.exists(pdf_filename):
            print(f"Error: File '{pdf_filename}' does not exist.")
            return

        # ���ļ���
        new_name = f"{new_text}.pdf"
        # ��ԭ���ļ�����һ�ݲ�����������new_text.pdf
        # shutil.copy(pdf_filename, new_name)

        # �������ļ�
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


# ʾ���÷�:
if __name__ == "__main__":
    # ������ ��ת�����ļ����ڵ��ļ���Ŀ¼ ������ļ�ͬʱ�����ڸ��ļ����У�������ҳ��ķ�ʽ��������
    file_path = '/Users/okrio/Downloads/test_pdf_pagenumber/'# �����ת�����ļ����ڵ��ļ���Ŀ¼
    find_pdf(file_path)
    print('succeed!!!')