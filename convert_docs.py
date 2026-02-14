import os
import win32com.client

def convert_doc_to_docx(folder_path):
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False

    for file in os.listdir(folder_path):
        if file.endswith(".doc") and not file.endswith(".docx"):
            full_path = os.path.abspath(os.path.join(folder_path, file))
            new_file = full_path + "x"  # adiciona x para virar .docx
            
            print(f"Convertendo: {file}")
            
            doc = word.Documents.Open(full_path)
            doc.SaveAs(new_file, FileFormat=16)  # 16 = docx
            doc.Close()

    word.Quit()

if __name__ == "__main__":
    convert_doc_to_docx("convencoes coletivas")
