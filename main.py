from flask import Flask, request, render_template, jsonify
from app import app
import time
import torch
from PIL import Image
from io import BytesIO
import io
from diffusers import EulerAncestralDiscreteScheduler, AutoPipelineForImage2Image
from diffusers.utils import load_image
import base64

token = app.config['TOKEN']

torch.cuda.empty_cache()

# pipe = AutoPipelineForImage2Image.from_pretrained("stablediffusionapi/disney-pixar-cartoon", torch_dtype=torch.float16, variant="fp16", use_auth_token=token)
# pipe = AutoPipelineForImage2Image.from_pretrained("rossiyareich/aniflatmixAnimeFlatColorStyle_v20-fp16", torch_dtype=torch.float16, variant="fp16", use_auth_token=token)

# pipe = AutoPipelineForImage2Image.from_pretrained(
#    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)

pipe = AutoPipelineForImage2Image.from_pretrained("danbrown/RevAnimated-v1-2-2", torch_dtype=torch.float16, variant="fp16", use_auth_token=token)

pipe = pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

@app.route('/')
def upload_form():
    return render_template('main.html')

@app.route('/img2img', methods=['POST'])
def upload_file():
    try:
        # check if the post request has the file part
        if 'selectImage' not in request.files:
            resp = jsonify({'message' : 'No file part in the request', 'error':''})
            resp.status_code = 400
            return resp

        inputImg = request.files.getlist('selectImage')[0]

        # prompt = request.files.getlist('prompt')
        prompt = "animal, cute, high quality, disney, pixar style, 8k, 3d animation"
        negative_prompt = "bad art, amateur, lowres, bad anatomy, disfigured face, cropped, worst quality, low quality, normal quality, extra digits, extra legs, fewer digits, fewer legs, ugly, watermark, text, error, nsfw, deformed, disfigured,oversaturated,grain, low-res, blurry, low resolution, cropped, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, dehydrated bad proportions, extra limbs, cloned face, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

        init_image = load_image(Image.open(inputImg).convert("RGB"))
        
        image = pipe(prompt, 
                negative_prompt=negative_prompt,
                image=init_image,
                # num_inference_steps = 20,
                strength = 0.4,
                ).images[0]


        buffered = io.BytesIO()
        image.save(buffered, format='png')
        img_str = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        errors = {}
        
        # 이미지가 검정색인지 검사
        decoded_img = Image.open(io.BytesIO(base64.b64decode(img_str.split(',')[1])))
        is_black_image = all(pixel == (0, 0, 0) for pixel in decoded_img.getdata())

        
        if is_black_image:
            success = False
            return jsonify({'message': '','error': 'Sorry, an error occurred. Please try again.'})
        else:
            success = True

        if success:
            return jsonify({'message': 'Files successfully changed', 'img_str': img_str, 'error': ''}), 201
        else:
            return jsonify(errors), 400

    
    except Exception as e:
        print(str(e))
        return jsonify({'message': '','error': 'Sorry, an error occurred. Please try again.'}), 500

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
