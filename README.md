# SAE error probing experiment

Tim Hua

You can use conda to get my environment 

```bash
conda env create -f requirements.yaml
```

I got the data from: https://github.com/saprmarks/geometry-of-truth/tree/rax/datasets

And also from this paper: https://arxiv.org/pdf/2502.16681
- Via their dropbox: https://www.dropbox.com/scl/fo/lvajx9100jsy3h9cvis7q/ACU8osTw0FCM_X-d8Wn-3ao/cleaned_data?dl=0&rlkey=tq7td61h1fufm01cbdu2oqsb5&subfolder_nav_tracking=1

The last probing experiment I ran. Didn't really work so now we're gonna give up. 

```bash
python sae_error_probing_ami/logistic_regression_probe.py  --dataset 114_nyc_borough_Manhattan.csv  149_twt_emotion_happiness.csv --result_suffix man_borough twt_happy --n_seeds 100
```