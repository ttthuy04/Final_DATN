"""
Web demo bao ve — Upload video -> nhan dien hanh vi -> video ket qua + bang thong ke.

Chay:
  cd Student_Behavior_Recognition_v3
  python web/app.py

Mo trinh duyet: http://127.0.0.1:7860
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from web.demo_runner import get_sample_video_path, run_demo_from_upload

STATS_HEADERS = ["Hanh vi (VI)", "Ma lop", "So luong", "Ty le"]

CSS = """
.gradio-container { max-width: 1100px !important; margin: auto; }
.title-block { text-align: center; margin-bottom: 0.5rem; }
"""


def process_video(video_file, progress=gr.Progress()):
    def on_progress(frac: float, msg: str) -> None:
        progress(frac, desc=msg)

    output_path, table, status = run_demo_from_upload(video_file, progress_callback=on_progress)
    if output_path is None:
        return None, [], status
    return output_path, table, status


def load_sample():
    sample = get_sample_video_path()
    if sample is None:
        return None, [], "Khong co video mau trong data/raw/ (test1.mp4 hoac classroom_test.mp4)."
    return sample, [], f"Da chon video mau: {Path(sample).name}. Bam 'Chay demo' de bat dau."


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Nhan dien hanh vi sinh vien") as app:
        gr.Markdown(
            """
            # Nhan dien hanh vi sinh vien trong lop hoc
            **Pipeline:** YOLO11 → MediaPipe **(25 khop khung xuong)** → ST-GCN (4 lop hanh vi)

            Upload video lop hoc, he thong tra ve video co **khung xuong + nhan** va bang thong ke.
            """,
            elem_classes=["title-block"],
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_video = gr.Video(
                    label="Upload video lop hoc",
                    sources=["upload"],
                    format="mp4",
                )
                with gr.Row():
                    run_btn = gr.Button("Chay demo", variant="primary", scale=2)
                    sample_btn = gr.Button("Dung video mau", scale=1)

                status_box = gr.Textbox(
                    label="Trang thai",
                    interactive=False,
                    lines=2,
                    placeholder="Upload video roi bam 'Chay demo'. Xu ly co the mat vai phut (CPU).",
                )

                gr.Markdown(
                    """
                    **4 lop hanh vi:** Binh thuong | Ngu gat | Dung day | Dung dien thoai

                    **Video ket qua:** box + **khung xuong do/cam** + nhan hanh vi

                    **Luu y:** Do chinh xac tren video lop nhieu nguoi co the thap hon test set (~87%)
                    """
                )

            with gr.Column(scale=1):
                output_video = gr.Video(label="Video ket qua", interactive=False)
                stats_table = gr.Dataframe(
                    headers=STATS_HEADERS,
                    label="Bang thong ke hanh vi",
                    interactive=False,
                    wrap=True,
                )

        run_btn.click(
            fn=process_video,
            inputs=[input_video],
            outputs=[output_video, stats_table, status_box],
        )
        sample_btn.click(
            fn=load_sample,
            outputs=[input_video, stats_table, status_box],
        )

    return app


def main() -> None:
    demo = build_app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        css=CSS,
    )


if __name__ == "__main__":
    main()
