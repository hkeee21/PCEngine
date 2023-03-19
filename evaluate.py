import argparse
import os
import random
import sys
import time
import warnings
from collections import defaultdict
from distutils.util import strtobool

import numpy as np
import torch
from scipy.stats import gmean
from tabulate import tabulate
from tqdm import tqdm

from dataset_build import make_dataset
from model_build import make_model
from configs.config import Config
from script.utils import sparse_collate_fn, build_conv_buffer


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def get_config(dataset):
    if dataset == "semantic_kitti":
        config_file = f"configs/default/semantic_kitti/default.yaml"
    elif dataset == "waymo":
        config_file = f"configs/default/waymo/centerpoint_default.yaml"
    elif dataset == "nuscenes":
        config_file = f"configs/default/nuscenes/centerpoint_default.yaml"
    elif dataset == "nuscenes_lidarseg":
        config_file = f"configs/default/nuscenes_lidarseg/default.yaml"
    else:
        raise NotImplementedError

    return config_file


@torch.no_grad()
def main(configs) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = False

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    BENCHMARKS = [
        ("semantic_kitti", "SemanticKITTI (1x width) segmentation"),
        ("semantic_kitti", "SemanticKITTI (0.5x width) segmentation"),
        ("nuscenes_lidarseg", "nuScenes-LiDARSeg (1 frame) segmentation"),
        ("nuscenes_lidarseg", "nuScenes-LiDARSeg (3 frames) segmentation"),
        ("nuscenes", "nuScenes detection"),
        ("waymo", "Waymo (1 frame) detection"),
        ("waymo", "Waymo (3 frames) detection"),
    ]
    PRECISION = ["fp32"]

    results = defaultdict(lambda: defaultdict(list))
    ratios = defaultdict(list)
    results_folder_prefix = "results" if not args.fast else "results_fast"

    for t, (d, benchmark) in enumerate(tqdm(BENCHMARKS, leave=False)):
        for precision in tqdm(
            PRECISION, desc=f"Benchmark: {benchmark}", leave=False
        ):
            config_file = get_config(d)
            configs.reload(config_file, recursive=True)
            configs.model.cr = 1.0
            configs.model.enable_fp16 = True if precision == "fp16" else False

            if d == "semantic_kitti":
                configs.model.cr = 1.0 if benchmark == "SemanticKITTI (1x width) segmentation" else 0.5
            elif d == "nuscenes_lidarseg":
                configs.dataset.max_sweeps = 1 if benchmark == "nuScenes-LiDARSeg (1 frame) segmentation" else 3
            elif d == "waymo":
                configs.dataset.max_sweeps = 1 if benchmark == "Waymo (1 frame) detection" else 3

            n_frame = configs.dataset.get("max_sweeps", 1)
            # mm_types = "adaptive groups" if configs.conv == 3 else "baseline"

            save_path = f"{results_folder_prefix}/times/{configs.dataset.name}_{configs.model.cr}_{n_frame}/PCEngine_{precision}.npy"
            if not args.restart and os.path.exists(save_path):
                tqdm.write("Saved results found, resuming...")
                times = np.load(save_path)
            else:
                dataset = make_dataset(configs, 100 if args.fast else -1)
                dataflow = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=1,
                    # num_workers=configs.workers_per_gpu,
                    pin_memory=False,
                    collate_fn=sparse_collate_fn,
                    shuffle=False
                )

                model = make_model(configs, dataset).cuda()
                for key, module in model.named_modules():
                    if "batchnorm" in module.__class__.__name__.lower():
                        module.forward = lambda x: x
                model.eval()

                enable_fp16 = configs.model.enable_fp16
                data_type = torch.float
                if enable_fp16:
                    data_type = torch.half
                    model = model.half()


                # tiled_scatter_gather = configs.model.get("tiled_scatter_gather", False)
                # if configs.conv == 3:
                    # assert enable_fp16
                #     conv_config_fn = f"group_configs/{configs.dataset.name}_{n_frame}_{configs.model.name}_{configs.model.cr}_fp16_{configs.hardware}_configs.npy"
                #     if not os.path.exists(conv_config_fn):
                #         print("profiling best config for each layer...")
                #         configs_path = get_config(d, "torchsparse")
                #         os.system(
                #             f"OMP_NUM_THREADS=1 python group_profile.py {configs_path} --model.enable_fp16 True --model.cr {configs.model.cr} \
                #             --hardware {configs.hardware} --dataset.max_sweeps {n_frame} --model.tiled_scatter_gather {tiled_scatter_gather}"
                #         )
                #     conv_configs = np.load(conv_config_fn, allow_pickle=True).item()

                # for key, module in model.named_modules():
                #     if isinstance(module, spnn.Conv3d):
                #         if configs.conv == 0:
                #             module.config = dict(
                #                 epsilon=0,
                #                 mm_thresh=0,
                #                 kmap_mode=configs.model.kmap_mode,
                #                 tiled_scatter_gather=tiled_scatter_gather,
                #             )
                #         elif configs.conv == 3:
                #             if key in conv_configs:
                #                 module.config = dict(
                #                     epsilon=conv_configs[key]["epsilon"],
                #                     mm_thresh=conv_configs[key]["mm_thresh"],
                #                     kmap_mode=configs.model.kmap_mode,
                #                     tiled_scatter_gather=tiled_scatter_gather,
                #                 )
                #             else:
                #                 module.config = dict(
                #                     epsilon=0,
                #                     mm_thresh=0,
                #                     kmap_mode=configs.model.kmap_mode,
                #                     tiled_scatter_gather=tiled_scatter_gather,
                #                 )

                cinfo = dict()
                cinfo["in"] = [256]
                cinfo["out"] = [256]
                cinfo["kernel"] = [2, 3] 
                buffer = build_conv_buffer(cinfo, 50000, data_type, "cuda")

                times = []
                for i, feed_dict in enumerate(
                    tqdm(
                        dataflow,
                        desc=f"precision: {precision}",
                        leave=False,
                    )
                ):
                    inputs = {}
                    for key, val in feed_dict.items():
                        if "name" in key:
                            continue
                        if hasattr(val, "cuda"):
                            val = val.cuda()
                        inputs[key] = val

                    if enable_fp16:
                        inputs["pts_input"].F = inputs["pts_input"].F.half()

                    
                    # if hasattr(inputs["pts_input"], "build_buffer"):
                    #     inputs["pts_input"].build_buffer(
                    #         4000000 * 64,
                    #         torch.half if enable_fp16 else torch.float,
                    #     )

                    tag = configs.unet_tag
                    if d == "waymo" or d == "nuscenes":
                        tag = configs.resnet_tag

                    inputs["pts_input"].buffer = buffer
                    inputs["pts_input"].init_tag = tag

                    # print(inputs["pts_input"].feats.shape)
                    # break

                    # _ = model(inputs)

                    with torch.cuda.amp.autocast(enabled=enable_fp16):
                        if i == 0:
                            for _ in range(10):
                                _ = model(inputs)
                                # if backend == "torchsparse":
                                inputs["pts_input"].cbook.clear()
                                inputs["pts_input"].kmaps.clear()
                                inputs["pts_input"].init_tag = tag

                        start_time = cuda_time()
                        # torch.cuda.cudart().cudaProfilerStart()
                        _ = model(inputs)
                        # torch.cuda.cudart().cudaProfilerStop()
                        times.append(cuda_time() - start_time)

            fps = 1 / np.mean(times)
            # if backend == "torchsparse":
            times_ts = times
            ratios["PCEngine"].append(1)
            results[benchmark][f"PCEngine ({precision})"].append(
                f"1.00 ({fps:.4f} FPS)"
            )
            # elif backend == "spconv":
            #     ratio = np.mean(times_ts) / np.mean(times)
            #     ratios[f"SPConv ({precision})"].append(ratio)
            #     results[benchmark][f"SPConv ({precision})"].append(
            #         f"{ratio:.2f} ({fps:.1f} FPS)"
            #     )
            # elif backend == "ME":
            #     ratio = np.mean(times_ts) / np.mean(times)
            #     ratios[f"MinkowskiEngine ({precision})"].append(ratio)
            #     results[benchmark][f"MinkowskiEngine ({precision})"].append(
            #         f"{ratio:.2f} ({fps:.1f} FPS)"
            #    )

            filename = f"{configs.dataset.name}_{configs.model.name}_result.txt"

            os.makedirs(f"{results_folder_prefix}/summaries", exist_ok=True)
            with open(f"{results_folder_prefix}/summaries/{filename}", "a+") as fd:
                fd.write(
                    f"PCEngine,"
                    + f"{n_frame} frame,"
                    + f"{configs.model.cr},"
                    + f"{precision},"
                    + f"{np.sum(times):.4f}s ± {0:.4f}s,"
                    + f"{np.mean(times)*1000:.4f}ms per sample,\n"
                )

            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            np.save(save_path, times)

        tqdm.write("")
        tqdm.write(f"Evaluation results on {benchmark}:")
        tqdm.write(tabulate(results[benchmark], headers="keys", tablefmt="grid"))

    # results_all = defaultdict(list)
    # if len(results) > 0:
    #     for key in results:
    #         results_all["Task"].append(key)
    #         for benchmark in results[key]:
    #             results_all[benchmark] += results[key][benchmark]
    #     results_all["Task"].append("Geometric Mean")
    #     for benchmark in results[key]:
    #         if benchmark == "TorchSparse (fp16)":
    #             results_all[benchmark].append("1.00")
    #         else:
    #             results_all[benchmark].append(f"{gmean(ratios[benchmark]):.2f}")

    #     tqdm.write("")
    #     tqdm.write("Final evaluation results on all benchmarks:")
    #     tqdm.write(tabulate(results_all, headers="keys", tablefmt="grid"))
    #     tqdm.write("See table 1 and table 2 in the artifact evaluation instruction to validate results.")
    # else:
    #     tqdm.write("")
    #     tqdm.write("No evaluation results are found!")


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"

    # SITES = {
    #     "KITTI": "http://www.cvlibs.net/datasets/kitti/user_register.php",
    #     "NuScenes": "https://nuscenes.org/sign-up?prevpath=nuscenes&prevhash=download",
    #     "Waymo Open": "https://waymo.com/open/licensing/",
    # }
    # for name, url in SITES.items():
    #      if not user_prompt(f"Have you registered at the website of {name} dataset?"):
    #         print(f"Please register at the website of {name} dataset ({url})!")
    #         sys.exit(0)
    
    configs = Config()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(configs)