# XOR experiment

Tim Hua

You can use conda to get my environment 

```bash
conda env create -f requirements.yaml
```

I got the data from: https://github.com/saprmarks/geometry-of-truth/tree/rax/datasets

And also from this paper: https://arxiv.org/pdf/2502.16681
- Via their dropbox: https://www.dropbox.com/scl/fo/lvajx9100jsy3h9cvis7q/ACU8osTw0FCM_X-d8Wn-3ao/cleaned_data?dl=0&rlkey=tq7td61h1fufm01cbdu2oqsb5&subfolder_nav_tracking=1

The two main files one can use are `train_probe` and `logistic_regression_probe`. 

Here's an example generating the llama results

```bash
python sae_error_probing_ami/train_probe.py --model meta-llama/Llama-3.1-8B --sae_release llama_scope_lxr_8x --sae_id l19r_8x  --dataset 114_nyc_borough_Manhattan.csv  149_twt_emotion_happiness.csv  155_athlete_sport_basketball.csv  headline_frontpage_sample.csv  all_cities.csv --result_suffix man_borough twt_happy ath_basketball headline_fp truth --n_seeds 120
```

You can use a similar command for the logistic regression. 

```bash
python sae_error_probing_ami/logistic_regression_probe.py  --dataset 114_nyc_borough_Manhattan.csv  149_twt_emotion_happiness.csv headline_frontpage_sample.csv  --result_suffix man_borough twt_happy headline_fp --n_seeds 100
```

`probe_training_steering_old.py` contains all of the probe training and steering code that generates results in the writeup. 

A lot of the data analysis is done using the R script in the notebooks folder. 

