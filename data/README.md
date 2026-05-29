# 数据文件说明

本文件夹包含复现神经网络实验所需的 4 个处理后输入数据文件：

- `runoff-Volume.xlsx`：16 个子流域逐日 GWLF 地表径流产流量。
- `groundwater-Volume.xlsx`：16 个子流域逐日 GWLF 地下潜流/基流产流量。
- `Flow.xlsx`：寸滩水文站逐日实测出口流量。
- `WatershedInfo_new.xlsx`：子流域属性与河网拓扑信息。

这些文件是公开代码直接读取的模型输入数据。原始《中华人民共和国水文年鉴》资料、气象站原始观测、土地利用栅格、DEM 数据和 GIS 预处理中间成果未在本仓库中再分发。原始数据来源和预处理流程请参见论文正文。

数据文件的 SHA256 校验值见 `CHECKSUMS_SHA256.txt`。

在仓库根目录运行完整实验矩阵：

```bash
python run_experiments.py --data-dir data --result-root results_final
```
