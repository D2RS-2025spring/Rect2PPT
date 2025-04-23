@echo off
echo 正在创建 conda 环境 "rect2PPT"...
conda create --name rect2PPT python=3.8 -y

echo 正在安装依赖包到 "rect2PPT" 环境...
conda run -n rect2PPT python -m pip install -r requirements.txt

echo.
echo 环境配置完成！你可以通过运行“conda activate rect2PPT”来使用此环境。
pause

