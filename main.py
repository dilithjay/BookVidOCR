from pipeline import Pipeline


pipeline = Pipeline(
    vid_path="/home/dilith/Projects/DocumentAI/data/eragon/eragon_book_flip_1.mp4",
    img_size=(1920, 1080),
    seg_model_dir="/home/dilith/Projects/DocumentAI/BookVidOCR/models/",
    frame_dir="/home/dilith/Projects/DocumentAI/data/eragon/images/",
    masked_dir="./masked_data/"
)

# pipeline.mask_frames()
pipeline.compute_performance_labels()
