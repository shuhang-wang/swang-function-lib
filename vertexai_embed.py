# https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/multimodal-embeddings#aiplatform_sdk_text_image_embedding-python_vertex_ai_sdk

import vertexai

from vertexai.vision_models import Image, MultiModalEmbeddingModel, Video

# TODO(developer): Update values for project_id,
#            image_path, video_path, contextual_text, video_segment_config
project_id = 'project-ml-training-prod'
vertexai.init(project=project_id, location="us-central1")

model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
image_path = '/home/shuhangwang/Documents/Code/swang_lib/dog.jpeg'
image = Image.load_from_file(image_path)
# video = Video.load_from_file(video_path)

embeddings = model.get_embeddings(
    image=image,
    # video=video,
    # video_segment_config=video_segment_config,
    # contextual_text=contextual_text,
)

print(f"Image Embedding: {embeddings.image_embedding}")
x = embeddings.image_embedding
import pdb; pdb.set_trace()

# # Video Embeddings are segmented based on the video_segment_config.
# print("Video Embeddings:")
# for video_embedding in embeddings.video_embeddings:
#     print(
#         f"Video Segment: {video_embedding.start_offset_sec} - {video_embedding.end_offset_sec}"
#     )
#     print(f"Embedding: {video_embedding.embedding}")

# print(f"Text Embedding: {embeddings.text_embedding}")