# fairbench/datasets/celebA.py
import torch
from torch.utils.data import DataLoader, random_split
# import tensorflow_datasets as tfds

import torchvision.transforms as T

def _build_transforms(img_size=128):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        # 표준화 값은 CelebA 커스텀(간단하게 0.5/0.5)
        T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])

class CelebASubset(torch.utils.data.Dataset):
    def __init__(self, tfds_split, transform, target_attr="Smiling",
                 sensitive_attrs=("Male","Young")):
        self.ds = tfds_split
        self.tf = transform
        self.target_attr = target_attr
        self.sensitive_attrs = sensitive_attrs

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex["image"].convert("RGB")
        x = self.tf(img)
        # target: Smiling(1/0)
        y = int(ex["attributes"][self.target_attr])
        # S: 여러 민감 속성
        S = {k:int(ex["attributes"][k]) for k in self.sensitive_attrs}
        return x, y, S

def load_celebA(args):
    """
    CelebA 로더 (TFDS)
      - GCS 미러 우선: tfds.load(..., try_gcs=True)
      - 수동 파일 지원: --celebA_manual_dir (해당 폴더에 identity_CelebA.txt 등 넣어두면 사용)
      - TFDS 캐시 경로: --tfds_data_dir
      - 체크섬 무시 옵션: --tfds_disable_checksums (가능하면 쓰지 말 것)
      - 민감 속성: --sens_keys (없거나 'auto'면 기본 4개)
      - 이미지 크기: --img_size (기본 128)
    """
    import os
    import tensorflow_datasets as tfds
    from torch.utils.data import DataLoader

    # ---- 인자/설정 ----
    data_dir   = getattr(args, "tfds_data_dir", None)           # TFDS 캐시 디렉터리
    manual_dir = getattr(args, "celebA_manual_dir", None)       # 수동 파일 디렉터리(있으면 사용)
    disable_cs = bool(getattr(args, "tfds_disable_checksums", False))
    img_size   = getattr(args, "img_size", 128)

    # 다운로드 설정 (GCS 우선 + manual_dir 지원)
    try:
        dl_cfg = tfds.download.DownloadConfig(
            try_download_gcs=True,
            manual_dir=manual_dir,
        )
    except TypeError:
        # 아주 구버전용 백업: 필드를 직접 세팅
        dl_cfg = tfds.download.DownloadConfig()
        try:
            dl_cfg.try_download_gcs = True
            dl_cfg.manual_dir = manual_dir
        except Exception:
            pass

    # ---- 다운로드/로딩: builder() 대신 load()로 GCS 경로 강제 ----
    try:
        split_dict = {"train": "train", "validation": "validation", "test": "test"}
        ds_dict, info = tfds.load(
            "celeb_a",
            split=split_dict,
            with_info=True,
            try_gcs=True,                         # ★ GCS 미러 우선
            data_dir=data_dir,
            download=True,
            download_and_prepare_kwargs={"download_config": dl_cfg},
            as_supervised=False,
            shuffle_files=False,
        )
    except Exception as e:
        # 가장 흔한 케이스: Google Drive checksum mismatch
        print(f"[ERROR] TFDS celeb_a load failed: {e}")
        print("        - 먼저 잘못 받은 파일들을 지우고 다시 시도하세요:")
        print("          rm -f ~/tensorflow_datasets/downloads/ucexport_download_id_1_ee_*")
        print("          rm -rf ~/tensorflow_datasets/downloads/extracted/*")
        print("        - 그래도 안 되면 --celebA_manual_dir 로 수동 파일 경로를 지정하세요.")
        print("          (img_align_celeba.zip, list_attr_celeba.txt, identity_CelebA.txt, list_eval_partition.txt 등)")
        print("        - 최후 수단: --tfds_disable_checksums 1 (권장X)")
        raise

    # ---- 전처리/민감속성 선택 ----
    tfm = _build_transforms(img_size=img_size)

    default_sens = ("Male", "Young", "Blond_Hair", "Heavy_Makeup")
    if getattr(args, "sens_keys", None) and str(args.sens_keys).lower() != "auto":
        sens = tuple([s.strip() for s in str(args.sens_keys).split(",") if s.strip()])
    else:
        sens = default_sens

    target = getattr(args, "celebA_target", "Smiling")

    # ---- PyTorch Dataset 래퍼 ----
    ds_train = ds_dict["train"]
    ds_valid = ds_dict["validation"]
    ds_test  = ds_dict["test"]

    train = CelebASubset(ds_train, tfm, target_attr=target, sensitive_attrs=sens)
    valid = CelebASubset(ds_valid, tfm, target_attr=target, sensitive_attrs=sens)
    test  = CelebASubset(ds_test,  tfm, target_attr=target, sensitive_attrs=sens)

    # ---- DataLoader ----
    dl_train = DataLoader(train, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    dl_valid = DataLoader(valid, batch_size=args.batch_size, shuffle=False, num_workers=2)
    dl_test  = DataLoader(test,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    meta = dict(
        source="tfds:celeb_a",
        target=target,
        sens_list=list(sens),
        img_size=img_size,
        tfds_data_dir=data_dir,
        celebA_manual_dir=manual_dir,
        tfds_disable_checksums=disable_cs,
    )
    return dict(train_loader=dl_train, val_loader=dl_valid, test_loader=dl_test, meta=meta, type="image")

