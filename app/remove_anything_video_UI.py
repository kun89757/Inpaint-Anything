import os
import sys
import cv2
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
os.chdir("../")
import gradio as gr
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import imageio_ffmpeg as ffmpeg
import remove_anything_video
import torch
import tempfile
# from omegaconf import OmegaConf
# from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama, build_lama_model, inpaint_img_with_builded_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
import argparse

def demo(model):
    def setup_args(parser):
        parser.add_argument(
            "--lama_config", type=str,
            default="./lama/configs/prediction/default.yaml",
            help="The path to the config file of lama model. "
                 "Default: the config of big-lama",
        )
        parser.add_argument(
            "--lama_ckpt", type=str,
            default="pretrained_models/big-lama",
            help="The path to the lama checkpoint.",
        )
        parser.add_argument(
            "--sam_ckpt", type=str,
            default="./pretrained_models/sam_vit_h_4b8939.pth",
            help="The path to the SAM checkpoint to use for mask generation.",
        )
    def mkstemp(suffix, dir=None):
        fd, path = tempfile.mkstemp(suffix=f"{suffix}", dir=dir)
        os.close(fd)
        return Path(path)


    def get_sam_feat(img):
        model['sam'].set_image(img)
        features = model['sam'].features
        orig_h = model['sam'].orig_h
        orig_w = model['sam'].orig_w
        input_h = model['sam'].input_h
        input_w = model['sam'].input_w
        model['sam'].reset_image()
        return features, orig_h, orig_w, input_h, input_w


    def get_masked_img(img, w, h, features, orig_h, orig_w, input_h, input_w, dilate_kernel_size):
        point_coords = [w, h]
        point_labels = [1]

        model['sam'].is_image_set = True
        model['sam'].features = features
        model['sam'].orig_h = orig_h
        model['sam'].orig_w = orig_w
        model['sam'].input_h = input_h
        model['sam'].input_w = input_w

        # model['sam'].set_image(img) # todo : update here for accelerating
        print(point_coords)
        masks, _, _ = model['sam'].predict(
            point_coords=np.array([point_coords]),
            point_labels=np.array(point_labels),
            multimask_output=True,
        )

        masks = masks.astype(np.uint8) * 255

        # dilate mask to avoid unmasked edge effect
        if dilate_kernel_size is not None:
            masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]
        else:
            masks = [mask for mask in masks]

        figs = []
        for idx, mask in enumerate(masks):
            # save the pointed and masked image
            tmp_p = mkstemp(".png")
            dpi = plt.rcParams['figure.dpi']
            height, width = img.shape[:2]
            fig = plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
            plt.imshow(img)
            plt.axis('off')
            show_points(plt.gca(), [point_coords], point_labels,
                        size=(width*0.04)**2)
            show_mask(plt.gca(), mask, random_color=False)
            plt.tight_layout()
            plt.savefig(tmp_p, bbox_inches='tight', pad_inches=0)
            figs.append(fig)
            plt.close()
        return *figs, *masks


    def get_inpainted_img(vid, x, y,mask):
        args = [
            '--input_video', str(vid),
            '--coords_type', 'key_in',
            '--point_coords', str(x), str(y),
            '--point_labels', str(mask),
            '--dilate_kernel_size', '15',
            '--output_dir', './results',
            '--sam_model_type', 'vit_b',
            '--sam_ckpt', './pretrained_models/sam_vit_b_01ec64.pth',
            '--lama_config', 'lama/configs/prediction/default.yaml',
            '--lama_ckpt', './pretrained_models/big-lama',
            '--tracker_ckpt', 'vitb_384_mae_ce_32x4_ep300',
            '--vi_ckpt', './pretrained_models/sttn.pth',
            '--mask_idx', '2',
            '--fps', '25'
        ]
        remove_anything_video.main(args)
        results_folder = "./results"
        files_in_folder = os.listdir(results_folder)
        removed_videos = [f for f in files_in_folder if f.startswith("removed")]
        if removed_videos:
            removed_video_path = os.path.join(results_folder, removed_videos[0])

        return removed_video_path

    # get args
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    # # build models
    # model = {}
    # # build the sam model
    # model_type="vit_h"
    # ckpt_p=args.sam_ckpt
    # model_sam = sam_model_registry[model_type](checkpoint=ckpt_p)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_sam.to(device=device)
    # model['sam'] = SamPredictor(model_sam)
    #
    # # build the lama model
    # lama_config = args.lama_config
    # lama_ckpt = args.lama_ckpt
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model['lama'] = build_lama_model(lama_config, lama_ckpt, device=device)

    with gr.Blocks() as demo:
        features = gr.State(None)
        orig_h = gr.State(None)
        orig_w = gr.State(None)
        input_h = gr.State(None)
        input_w = gr.State(None)

        with gr.Row().style(mobile_collapse=False, equal_height=True):
            with gr.Column(variant="panel"):
                with gr.Row():
                    gr.Markdown("## Input Video")
                with gr.Row():
                    video = gr.Video(label="Input Video").style(height="200px")
            with gr.Column(variant="panel"):
                with gr.Row():
                    gr.Markdown("## Input Image")
                with gr.Row():
                    img = gr.Image(label="Input Image").style(height="200px")
            with gr.Column(variant="panel"):
                with gr.Row():
                    gr.Markdown("## Pointed Image")
                with gr.Row():
                    img_pointed = gr.Plot(label='Pointed Image')
            with gr.Column(variant="panel"):
                with gr.Row():
                    gr.Markdown("## Control Panel")
                with gr.Row():
                    w = gr.Number(label="Point Coordinate W")
                    h = gr.Number(label="Point Coordinate H")
                dilate_kernel_size = gr.Slider(label="Dilate Kernel Size", minimum=0, maximum=100, step=1, value=15)
                mask_sel=gr.Radio(["0", "1", "2"], label="Mask", info="Select Mask")
                sam_mask = gr.Button("Predict Mask", variant="primary").style(full_width=True, size="sm")
                lama = gr.Button("Inpaint Image", variant="primary").style(full_width=True, size="sm")
                clear_button_image = gr.Button(value="Reset", label="Reset", variant="secondary").style(full_width=True, size="sm")


        def get_video_frame(vid):
            print(mask_sel)
            video_path = vid  # 视频文件路径
            print(vid)
            save_path = "pic.jpg"  # 保存路径及文件名
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            f_img = frame_rgb
            cv2.imwrite(save_path, frame_rgb)
            cap.release()
            img.value="pic.jpg"
            return f_img




        def get_first_frame(vid):
            video_path = vid  # 将 'your_video.mp4' 替换为你的视频文件路径
            with ffmpeg.get_reader(video_path) as video:
                first_frame = video.get_data(0)  # 获取第一帧
            img.value = first_frame
        video.change(get_video_frame, [video], [img]

                   )
        img.change(get_sam_feat, [img], [features, orig_h, orig_w, input_h, input_w])


        # todo: maybe we can delete this row, for it's unnecessary to show the original mask for customers
        with gr.Row(variant="panel"):
            with gr.Column():
                with gr.Row():
                    gr.Markdown("## Segmentation Mask")
                with gr.Row():
                    mask_0 = gr.outputs.Image(type="numpy", label="Segmentation Mask 0").style(height="200px")
                    mask_1 = gr.outputs.Image(type="numpy", label="Segmentation Mask 1").style(height="200px")
                    mask_2 = gr.outputs.Image(type="numpy", label="Segmentation Mask 2").style(height="200px")

        with gr.Row(variant="panel"):
            with gr.Column():
                with gr.Row():
                    gr.Markdown("## Image with Mask")
                with gr.Row():
                    img_with_mask_0 = gr.Plot(label="Image with Segmentation Mask 0")
                    img_with_mask_1 = gr.Plot(label="Image with Segmentation Mask 1")
                    img_with_mask_2 = gr.Plot(label="Image with Segmentation Mask 2")

        with gr.Row(variant="panel"):
            with gr.Column():
                with gr.Row():
                    gr.Markdown("## video Removed with Mask")
                with gr.Row():
                    out_video = gr.Video(label="Output Video").style(height="200px")


        def get_select_coords(img, evt: gr.SelectData):
            dpi = plt.rcParams['figure.dpi']
            height, width = img.shape[:2]
            fig = plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            show_points(plt.gca(), [[evt.index[0], evt.index[1]]], [1],
                        size=(width*0.04)**2)
            return evt.index[0], evt.index[1], fig

        img.select(get_select_coords, [img], [w, h, img_pointed])
        # img.change(get_sam_feat, [img], [features, orig_h, orig_w, input_h, input_w])

        sam_mask.click(
            get_masked_img,
            [img, w, h, features, orig_h, orig_w, input_h, input_w, dilate_kernel_size],
            [img_with_mask_0, img_with_mask_1, img_with_mask_2, mask_0, mask_1, mask_2]
        )

        lama.click(
            get_inpainted_img,
            [video, w, h,mask_sel],
            [out_video]
        )


        def reset(*args):
            return [None for _ in args]

        clear_button_image.click(
            reset,
            [video,img, features, img_pointed, w, h, mask_0, mask_1, mask_2, img_with_mask_0, img_with_mask_1, img_with_mask_2, ],
            [video,img, features, img_pointed, w, h, mask_0, mask_1, mask_2, img_with_mask_0, img_with_mask_1, img_with_mask_2, ]
        )
        return demo


if __name__ == "__main__":
    demo().launch(share=True)
    