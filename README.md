# XOR experiment

Tim Hua

You can use conda to get my environment 

```bash
conda env create -f requirements.yaml
```

I got the data from: https://github.com/saprmarks/geometry-of-truth/tree/rax/datasets

And also from this paper: https://arxiv.org/pdf/2502.16681
- Via their dropbox: https://www.dropbox.com/scl/fo/lvajx9100jsy3h9cvis7q/ACU8osTw0FCM_X-d8Wn-3ao/cleaned_data?dl=0&rlkey=tq7td61h1fufm01cbdu2oqsb5&subfolder_nav_tracking=1


`probe_training_main.py` contains all of the probe training and steering code that generates most of the results. It should run on one go but would likely take up more than 24 GB of VRAM when it gets to the steering part. So I run that part separately--you can skip all of the probe generation, as well as the patching experiment which I suspect takes up a bunch of ram and doesn't give it back--in VS code python interactive mode. 

Everything there should be super reproducible. 
