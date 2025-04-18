docker run -d --gpus all --rm -p 8085:80 \
-v $(pwd)/Meta-Llama-3-8B-Instruct:/data \
-e MODEL_ID=/data \
-e SAFETENSORS=true \
--name TGI_stream \
ghcr.io/huggingface/text-generation-inference:1.4.0 \
--port 80

