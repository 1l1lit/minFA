1. 克隆仓库
    - git clone https://gitcode.com/Riht5/FA.git
2. 进入仓库
    - cd FA
3. 配置虚拟环境
    - python -m venv venv --system-site-packages
4. 激活虚拟环境
    - source venv/bin/activate
5. 编译
    - python setup.py install
6. 测试
    - python run.py
    - pip install pytest -i https://mirrors.aliyun.com/pypi/simple/
    - pytest -q --tb=short tests/test_flash_attn_npu.py
7. 更新代码
    - git commit --amend --no-edit
    - git push --force-with-lease origin main
    - git pull --rebase origin main

git clone https://gitcode.com/Riht5/FA.git && cd FA && python -m venv venv --system-site-packages && source venv/bin/activate && python setup.py install

pip install pytest -i https://mirrors.aliyun.com/pypi/simple/ && pytest -q --tb=short tests/test_flash_attn_npu.py