import os
import argparse
from  minio import Minio
from typing import Optional, Union
from io import BytesIO
import zipfile

class kmc():

    client = Minio("10.200.100.66:9000", access_key="minioadmin", secret_key="miniopassword", secure=False)

    @classmethod
    def get_object(self, bucket, objects):
        response = self.client.get_object(bucket, objects)
        result = BytesIO(response.read())
        response.close()
        response.release_conn()
        return result

    @staticmethod
    def is_zip(filepath: Union[str, BytesIO]) -> bool:  # Optional → Union
        try:
            return zipfile.is_zipfile(filepath)
        except Exception:
            return False

    @staticmethod
    def _report_csv(result: bool, context: str) -> None:
        if result:
            print(f"found csv file in {context}")
        else:
            print(f"no csv file in {context}")

    @classmethod
    def download_object(
        cls,
        bucket: str,
        objects: str,
        dest_path: Optional[str] = None,
        overwrite: bool = False,
        create_dirs: bool = True,
        auto_extract: bool = False,
        extract_dir: Optional[str] = None
    ) -> str:
        """
        下載 MinIO 物件，可選擇下載後自動解壓縮 ZIP。

        Args:
            bucket       : bucket 名稱
            objects      : 物件 key
            dest_path    : 下載到本地的檔案路徑，預設同檔名
            overwrite    : 本地檔案存在時是否覆寫
            create_dirs  : 自動建立資料夾
            auto_extract : 若為 ZIP，下載完成後是否自動解壓
            extract_dir  : 解壓目的資料夾 (None → 與 dest_path 同一層、同名資料夾)

        Returns:
            dest_path    : 實際下載檔案的本地路徑
        """
        if dest_path is None:
            dest_path = os.path.basename(objects)

        # 先處理本地檔案存在的情況
        if os.path.exists(dest_path) and not overwrite:
            print(f"[SKIP] {dest_path} already exists.")
        else:
            # 確保資料夾存在
            if create_dirs:
                os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

            print(f"[DL] {bucket}/{objects}  →  {dest_path}")
            cls.client.fget_object(bucket, objects, dest_path)

        #── 檢查是否含 .csv ───────────────────
        if cls.is_zip(dest_path):
            with zipfile.ZipFile(dest_path, "r") as zf:
                has_csv = any(name.lower().endswith(".csv") for name in zf.namelist())
                cls._report_csv(has_csv, f"zip '{dest_path}'")
                # optional: raise if not has_csv
                # if not has_csv:
                #     raise ValueError("ZIP 檔內找不到 metadata")

        # ---------- 自動解壓 ----------
        if auto_extract and cls.is_zip(dest_path):
            if extract_dir is None:
                # 預設解壓到 <dest_path 去掉副檔名> 資料夾
                extract_dir = os.path.splitext(dest_path)[0]
            os.makedirs(extract_dir, exist_ok=True)

            print(f"[UNZIP] {dest_path}  →  {extract_dir}/")
            with zipfile.ZipFile(dest_path, "r") as zf:
                zf.extractall(path=extract_dir)

        return dest_path



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', default="jasonkuo", type=str)
    parser.add_argument('--objects', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    
    bucket = args.bucket
    objects = args.objects
    output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)

    print(f"bucket name = {bucket}",
          f"object file: {objects}")
    assert kmc.client.bucket_exists(bucket), f"bucket named {bucket} does not exist"
   
    kmc.download_object(
        bucket=bucket,
        objects=objects,
        auto_extract=True,          
        extract_dir=output_path
    )




 