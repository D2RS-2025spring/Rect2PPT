@echo off
echo 正在根据 environment_full.yaml 创建 conda 环境 "rect2PPT"...
conda env create -f environment.yaml

echo.
echo 环境配置完成！你可以通过运行“conda activate rect2PPT”来使用此环境。
pause


