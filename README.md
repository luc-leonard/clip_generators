# clip_generators
some text to image models, all using clips. Include bots for IRC and discord. This relies A LOT on the work of https://twitter.com/RiversHaveWings

you will need to setup a virtual env, and install pytorch 1.7.1 yourself, with the CUDA version that match your hardware
(cf. https://pytorch.org/get-started/previous-versions/ )

Once this is done, you can run `pip install -r requirements.txt`

Finally You will need to download the VQGAN pretrained model here: https://mega.nz/folder/KFVCSJSR#vsk_dW2wbPUu4mnt8SZvQA
Put the `imagenet` folder in `clip_generators/models/taming_transformers/models`
(TODO: find somewhere to direct link this, so the download could be automated)

you will need to run `export PYTHONPATH=$PWD` in the root folder of the repo before starting any script

To use locally, just run `python main.py "<your text here>"`
