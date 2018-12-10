# gramzoom

Zooming in on images that optimize the gram matrix from another image.

The project page for this repository can be found at:

https://douweosinga.com/projects/gramzoom

## Installation

To setup a virtual environment with python3 and the dependencies installed,
execute in a shell:

    python3 -m venv venv3
    source venv3/bin/activate
    pip install -r requirements.txt


You can now run the main script:

    python gramzoom.py --steps=660 \
                       --target_size=640 \
                       --zoom_from=10 \
                       --samples_per_step=7 \
                       --input=kittens.png \
                       --output=kittens.mp4

And it should create a zooming in movie:

![Kittens](kittens.gif)

## Usage

The command line parameters shouldn't surprise anybody. `steps` determines how many steps to
run the script. `target_size` determines how big the movie will be. `zoom_from` is a bit more
tricky, basically this determines the number of warm-up frames before we start zooming
and saving frames. `samples_per_step` determines how many times we run the algorithm per
frame - more results in slower progress but higher quality frames. Then there is the input
and output.
