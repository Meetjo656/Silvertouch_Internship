import os
import zipfile

src_dir = r"d:\Silvertouch Internship\task3\visualized_user_images"
out_zip = r"d:\Silvertouch Internship\task3\visualized_images.zip"

with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            file_path = os.path.join(root, file)
            zipf.write(file_path, arcname=file)

print("Zip created successfully at:", out_zip)
