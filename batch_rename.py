import os
import shutil

#添加r，让python忽略转义字符，不然 \r 是回车符
input_dir = r'F:\repo\rs-data-prep\records\phone\videos'
output_dir = r'/records/phone/videos'

os.makedirs(output_dir, exist_ok=True)
#指定文件后缀名
file_extension = ('.mp4','bag')
#新的命名
index = 1
for filename in os.listdir(input_dir):
    if filename.lower().endswith(file_extension):
        extension = os.path.splitext(filename)[1]
        #新文件名
        new_filename = f"{index}{extension}"
        old_path = os.path.join(input_dir,filename)
        new_path = os.path.join(output_dir,new_filename)
        #复制文件到新的位置
        shutil.copy2(old_path,new_path)
        print(f"文件{filename} 改名为 ---> {new_filename}")

        index += 1
print(f"处理完成，保存路径：{output_dir}")