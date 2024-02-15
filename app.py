import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

input_points = []
IMG_SIZE = 512
input_image = None

def generate_app(get_processed_inputs, inpaint):
    
    global input_points
    global input_image
    
    def get_points(img, evt: gr.SelectData):
        
        global input_image
        
        # The first time this is called, we save the untouched
        # input image
        if len(input_points) == 0:
            input_image = img.copy()
        
        x = evt.index[0]
        y = evt.index[1]

        input_points.append([x, y])
        
        # Run SAM
        sam_output = run_sam()
        
        # Mark selected points with a green crossmark
        draw = ImageDraw.Draw(img)
        size = 10
        for point in input_points:

            x, y = point

            draw.line((x - size, y, x + size, y), fill="green", width=5)
            draw.line((x, y - size, x, y + size), fill="green", width=5)
        
        return sam_output, img


    def run_sam():
        
        if input_image is None:
            raise gr.Error("No points provided. Click on the image to select the object to segment with SAM")
        
        try:
            
            mask = get_processed_inputs(input_image, [input_points])
            
            res_mask = np.array(Image.fromarray(mask).resize((IMG_SIZE, IMG_SIZE)))
            
            return (
                input_image.resize((IMG_SIZE, IMG_SIZE)), 
                [
                    (res_mask, "background"), 
                    (~res_mask, "subject")
                ]
            )
        except Exception as e:
            raise gr.Error(str(e))


    def run(prompt, negative_prompt, cfg, seed, invert):

        if input_image is None:
            raise gr.Error("No points provided. Click on the image to select the object to segment with SAM")
        
        amask = run_sam()[1][0][0]
        
        if bool(invert):
            what = 'subject'
            amask = ~amask
        else:
            what = 'background'
        
        gr.Info(f"Inpainting {what}... (this will take up to a few minutes)")
        try:
            inpainted = inpaint(input_image, amask, prompt, negative_prompt, seed, cfg)
        except Exception as e:
            raise gr.Error(str(e))

        return inpainted.resize((IMG_SIZE, IMG_SIZE))
    
    def reset_points(*args):
        
        input_points.clear()
    
    
    def preprocess(input_img):
        
        if input_img is None:
            return None
        
        # Make sure the image is square
        width, height = input_img.size
        
        if width != height:
            
            gr.Warning("Image is not square, adding white padding")

            # Determine the size for the new square image
            new_size = max(width, height)

            # Create a new image with the desired size and background color
            # Change 'black' to your desired background color if needed
            new_image = Image.new("RGB", (new_size, new_size), 'white')
            
            # Calculate the position to paste the original image onto the new image
            left = (new_size - width) // 2
            top = (new_size - height) // 2

            # Paste the original image onto the new image
            new_image.paste(input_img, (left, top))
            
            input_img = new_image
        
        return input_img.resize((IMG_SIZE, IMG_SIZE))
    
    
    with gr.Blocks() as demo:

        gr.Markdown(
        """
        # Image inpainting
        1. Upload an image by clicking on the first canvas.
        2. Click on the subject you would like to keep. Immediately SAM will be run and you will see the results. If you
           are happy with those results move to the next step, otherwise add more points to refine your mask.
        3. Write a prompt (and optionally a negative prompt) for what you want to generate for the infilling. 
           Adjust the CFG scale and the seed if needed. You can also invert the mask, i.e., infill the subject 
           instead of the background by toggling the relative checkmark.
        4. Click on "run inpaint" and wait for up to two minutes. If you are not happy with the result, 
           change your prompts and/or the settings (CFG scale, random seed) and click "run inpaint" again.

        # EXAMPLES
        Scroll down to see a few examples. Click on an example and the image and the prompts will be filled for you. 
        Note however that you still need to do step 2 and 4.
        """)

        with gr.Row():
            
            # This is what the user will interact with
            display_img = gr.Image(
                label="Input", 
                interactive=True, 
                type='pil',
                height=IMG_SIZE,
                width=IMG_SIZE
            )

            sam_mask = gr.AnnotatedImage(
                label="SAM result",
                interactive=False,
                height=IMG_SIZE,
                width=IMG_SIZE,
                color_map={"background": "#a89a00"}
            )

            result = gr.Image(
                label="Output",
                interactive=False,
                type='pil',
                height=IMG_SIZE,
                width=IMG_SIZE,
            )            
        
        # Events
        display_img.select(get_points, inputs=[display_img], outputs=[sam_mask, display_img])
        display_img.clear(reset_points)
        display_img.change(preprocess, inputs=[display_img], outputs=[display_img])
        
        with gr.Row():
            cfg = gr.Slider(
                label="Classifier-Free Guidance Scale", 
                minimum=0.0, 
                maximum=20.0, 
                value=7, 
                step=0.05
            )
            random_seed = gr.Number(
                label="Random seed", 
                value=74294536, 
                precision=0
            )
            checkbox = gr.Checkbox(
                label="Infill subject instead \nof background"
            )

        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt for infill"
            )
            neg_prompt = gr.Textbox(
                label="Negative prompt"
            )
            
            reset_points_b = gr.ClearButton(
                value="Reset", 
                components=[
                    display_img, 
                    sam_mask, 
                    result,
                    prompt,
                    neg_prompt,
                    checkbox
                ]
            )
            reset_points_b.click(reset_points)
            
            submit_inpaint = gr.Button(value="Run inpaint")

        with gr.Row():
            examples = gr.Examples(
                [
                    [
                        "car.png", 
                        "a car driving on planet Mars. Studio lights, 1970s", 
                        "artifacts, low quality, distortion",
                        74294536
                    ],
                    [
                        "dragon.jpeg",
                        "a dragon in a medieval village",
                        "artifacts, low quality, distortion",
                        97
                    ],
                    [
                        "monalisa.png",
                        "a fantasy landscape with flying dragons",
                        "artifacts, low quality, distortion",
                        97
                    ]
                ],
                inputs=[
                    display_img,
                    prompt,
                    neg_prompt,
                    random_seed
                ]

            )

        submit_inpaint.click(
            fn=run, 
            inputs=[
                prompt, 
                neg_prompt,
                cfg,
                random_seed,
                checkbox
            ], 
            outputs=[result]
        )

    demo.queue(max_size=1).launch(share=True, debug=True)
    
    return demo