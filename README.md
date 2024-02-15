# AI-Photo-Editing-with-Inpainting

## About
In this project I build a little app that allows you to select a subject and then change its background, OR keep the background and change the subject.

The process involves a user uploading an image and selecting the main object by clicking on it. The Segment Anything Model (SAM) is activated to create a mask around the selected object, choosing the most accurate mask generated. The user is shown this result to either accept it or refine the mask further with additional points. Once the mask is finalized, the user gives a text description (and possibly a negative prompt) to specify a new background for the selected object. An infill model then creates this new background, and the final image is displayed. Optionally, the user can choose to invert the mask and substitute the subject while keeping the background, as in the example above.

This little app can be used to swap backgrounds, swap subjects, remove objects, and more!

## Installation

If you are trying this locally, follow these instructions first:

```bash
conda create -n "impating" python=3.11

conda activate impating

git clone https://github.com/eljandoubi/AI-Photo-Editing-with-Inpainting.git

cd AI-Photo-Editing-with-Inpainting

pip install -r requirements.txt

jupyter notebook
```
