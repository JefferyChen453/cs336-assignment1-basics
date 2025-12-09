# uv run -m cs336_basics.transformer.train.main \
#     --config cs336_basics/config/train_tinystories.yaml \
#     --optim.lr 1e-5 \
#     --optim.min_lr 1e-6 \
#     --trainer.total_steps 1000 \
#     --trainer.experiment_name tinystories_lr_1e-5\
#     --trainer.save_freq 2000 \

# uv run -m cs336_basics.transformer.train.main \
#     --config cs336_basics/config/train_tinystories.yaml \
#     --optim.lr 5e-5 \
#     --optim.min_lr 5e-6 \
#     --trainer.total_steps 1000 \
#     --trainer.experiment_name tinystories_lr_5e_5\
#     --trainer.save_freq 2000 \

# uv run -m cs336_basics.transformer.train.main \
#     --config cs336_basics/config/train_tinystories.yaml \
#     --optim.lr 1e-4 \
#     --optim.min_lr 1e-5 \
#     --trainer.total_steps 1000 \
#     --trainer.experiment_name tinystories_lr_1e_4\
#     --trainer.save_freq 2000 \

# uv run -m cs336_basics.transformer.train.main \
#     --config cs336_basics/config/train_tinystories.yaml \
#     --optim.lr 5e-4 \
#     --optim.min_lr 5e-5 \
#     --trainer.total_steps 1000 \
#     --trainer.experiment_name tinystories_lr_5e_4\
#     --trainer.save_freq 2000 \

uv run -m cs336_basics.transformer.train.main \
    --config cs336_basics/config/train_tinystories.yaml \
    --optim.lr 2e-4 \
    --optim.min_lr 2e-5 \
    --trainer.total_steps 1000 \
    --trainer.experiment_name tinystories_lr_2e_4\
    --trainer.save_freq 2000 \

uv run -m cs336_basics.transformer.train.main \
    --config cs336_basics/config/train_tinystories.yaml \
    --optim.lr 3e-4 \
    --optim.min_lr 2e-5 \
    --trainer.total_steps 1000 \
    --trainer.experiment_name tinystories_lr_3e_4\
    --trainer.save_freq 2000 \
