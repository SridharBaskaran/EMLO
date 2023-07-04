# Pytorch Lightning - Hydra
## Development
### Install in dev mode
```
pip install -e .
pip install -r requirements.txt
```
```
mylib --help
mylib_train data.num_workers=16
mylib_train data.num_workers=16 trainer.deterministic=True +trainer.fast_dev_run=True
```

## Docker
```
docker build -f Dockerfile -t cifar-timm:latest .
docker run -it cifar-timm
mylib_train
```
