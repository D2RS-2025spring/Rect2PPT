import os
import sys
import yaml
import re
from subprocess import check_output
from invoke import task

def load_config(config_file="config.yaml"):
    """
    读取配置文件，返回配置字典。
    这里默认读取 "Paths" 分区下的配置项。
    """
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            try:
                config = yaml.safe_load(f)
                return config.get("Paths", {})
            except Exception as e:
                print("读取配置文件时出错:", e)
                sys.exit(1)
    else:
        print(f"配置文件 {config_file} 不存在。")
        return {}

# 以下为环境相关辅助方法
def get_env_name(yaml_file="environment.yaml"):
    if os.path.exists(yaml_file):
        with open(yaml_file, "r", encoding="utf-8") as f:
            try:
                env_data = yaml.safe_load(f)
                env_name = env_data.get("name")
                if env_name:
                    return env_name.strip()
                else:
                    print("Warning: environment.yaml 中未找到 'name' 键。")
            except Exception as e:
                print("Error parsing environment.yaml:", e)
                sys.exit(1)
    else:
        print("Warning: environment.yaml 文件不存在。")
    return None

def conda_env_exists(env_name):
    try:
        envs_output = check_output("conda env list", shell=True, encoding="utf-8")
        target = env_name.lower()
        for line in envs_output.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.replace("*", "").strip()
            match = re.match(r"^(\S+)", line)
            if match:
                candidate = match.group(1).strip()
                if os.path.sep in candidate:
                    candidate = os.path.basename(candidate)
                if candidate.lower() == target:
                    return True
        return False
    except Exception as e:
        print("无法检查 conda 环境:", e)
        return False

@task
def setup_env(c):
    """
    检查并创建或更新 Conda 环境，利用 environment.yaml 文件。
    """
    yaml_file = "environment.yaml"
    env_name = get_env_name(yaml_file)
    if not env_name:
        print("未能从 environment.yaml 获取环境名称。")
        sys.exit(1)
    print(f"目标 Conda 环境: {env_name}")
    current_env = os.environ.get("CONDA_DEFAULT_ENV")
    if current_env and current_env.lower() == env_name.lower():
        print("当前已在目标 Conda 环境中。")
        return
    if conda_env_exists(env_name):
        print(f"环境 '{env_name}' 已存在。")
        print(f"请运行: conda activate {env_name}")
        sys.exit(0)
    else:
        print(f"环境 '{env_name}' 不存在，正在创建...")
        result = c.run(f"conda env create -f {yaml_file}", warn=True)
        if result.failed:
            print("创建环境失败，请手动处理。")
            sys.exit(1)
        else:
            print(f"环境创建成功！请运行: conda activate {env_name}")
            sys.exit(0)

@task(help={
    "input": "输入图片（或图片目录）的路径；不传则从配置文件的 Paths.orig_folder 中读取",
    "tmp_dir": "存放裁剪图片的目录；不传则从配置文件的 Paths.mask_folder 中读取"
})
def run_detect(c, input=None, tmp_dir=None):
    """
    调用 detect_rect.py 模块，对图片进行识别与裁剪。
    优先使用命令行传入的参数，其次使用配置文件 Paths 分区中的设置。
    """
    config = load_config()
    # 使用配置文件中 Paths 分区的键名
    input = input or config.get("orig_folder")
    tmp_dir = tmp_dir or config.get("mask_folder")
    if not input or not tmp_dir:
        print("请通过命令行参数或配置文件指定 --input 和 --tmp_dir 参数。")
        sys.exit(1)
    cmd = f'python detect_rect.py --input "{input}" --tmp_dir "{tmp_dir}"'
    print(f"执行命令: {cmd}")
    c.run(cmd, echo=True)

@task(help={
    "image_dir": "存放裁剪后图片的目录；不传则从配置文件的 Paths.mask_folder 中读取",
    "output": "输出 PPT 文件保存目录；不传则从配置文件的 Paths.output_folder 中读取"
})
def run_create_ppt(c, image_dir=None, output=None):
    """
    调用 create_ppt.py 模块，将裁剪后的图片生成 PPT 文件。
    优先使用命令行传入的参数，其次使用配置文件 Paths 分区中的设置。
    """
    config = load_config()
    image_dir = image_dir or config.get("mask_folder")
    output = output or config.get("output_folder")
    if not image_dir or not output:
        print("请通过命令行参数或配置文件指定 --image_dir 和 --output 参数。")
        sys.exit(1)
    cmd = f'python create_ppt.py --image_dir "{image_dir}" --output "{output}"'
    print(f"执行命令: {cmd}")
    c.run(cmd, echo=True)

@task(pre=[setup_env, run_detect, run_create_ppt])
def all(c):
    """
    执行所有任务：
      1. 检查或创建 Conda 环境（setup_env）
      2. 运行图像处理模块（run_detect）
      3. 运行 PPT 生成模块（run_create_ppt）

    参数优先从命令行传入，其次使用配置文件 config.yaml 的 Paths 分区中的设置。
    """
    print("所有任务执行完成！")

if __name__ == "__main__":
    from invoke import Program
    program = Program(namespace=globals())
    program.run()
