# Anomaly Detection with Variance Stabilized Density Estimation

## Install
```
pip3 install -r req.txt
```

## Train
```
python3 main_many_seeds.py --num_seeds 3 --batch_size 4096 --learning_rate 1e-4 --dropout 0.1 --epochs 3000 --num_ensemble_models 3 --optimizer adam --variable_batch_size_factor 10 --ensemble_method isml --ll_loss_lambda 1.0  --var_loss_lambda 3.33333 --polyak_decay 0.995 --scheduler cos_annealing --device cuda:0,cuda:1 --num_parallel_procs 1 --hidden_dim 1024 --num_training_rounds 1 --nits_arch [16,16,2] --early_stopping --use_wandb --save_path ./results 
```

## Evaluate
```
python3 evaluate.py
```