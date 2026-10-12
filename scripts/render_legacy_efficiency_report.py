import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
records = json.loads((root/'summary_reviewed.json').read_text())
lines = ['# LPN、MCCG、MuSe-Net、InfoGeo效率评测', '',
 'dy-3090服务器GPU 7，RTX 3090 24GB，pyra_geo。2026-10-11完成。', '',
 '统一v2协议：FP32，TF32关闭，eval/inference；预热100次，每轮500次，共3轮。',
 'Latency为batch=1逐次同步墙钟计时；FPS为batch=16连续forward区间同步计时。',
 '范围是GPU输入图像到归一化描述子，排除读图、预处理、传输和检索。', '',
 '| 模型 | 输入 | Params (M) | FLOPs (G) | Latency B=1 (ms) | FPS B=16 | 峰值显存 B=1 (MiB) |',
 '|---|---|---:|---:|---:|---:|---:|']
for rec in records:
 r=rec['summary']
 lines.append(f"| {r['method']} | {r['resolution']}×{r['resolution']} | {r['params_m']:.3f} | {r['gflops']:.3f} | {r['latency_b1_mean_ms']:.2f} ± {r['latency_round_std_ms']:.2f} | {r['fps_b16']:.2f} | {r['peak_allocated_b1_mib']:.2f} |")
lines += ['', 'Latency是三轮均值±轮间标准差；全部轮次计入，未选择最快轮次。',
 'FLOPs采用1 MAC=2 FLOPs，非MAC运算使用声明的估计；参数统计实际执行路径。', '',
 '四个模型均严格加载、通过部署前后及offload后的描述子检查；FLOPs执行算子覆盖完整。',
 '此处的一致性检查证明统一测速入口与现有适配器一致，不证明兼容重建与作者实现一致。', '',
 '## 必须保留的限定', '',
 '- MCCG：公开源码兼容推理。checkpoint时代配套源码未确认，不能视为完全复现。',
 '- InfoGeo-RD-trial：重建的view-token DINOv2骨干及backbone→MixVPR路径；RD训练版本未确认。',
 '- MuSe-Net：net_209.pth；保留学习到的风格条件分支，上传配置含extra_Google=true。',
 '- LPN：net_119.pth；四部分描述子，stride=1；训练opts.yaml缺失，采用已审计默认配置。', '',
 '## 计数修正和来源', '',
 'MuSe-Net最近邻上采样只复制数据，计0浮点运算；已补充结构算子登记并通过测试。',
 '精确源码比较证明计时、融合和精度代码没有变化；本组四模型按补齐后的计数器完整复测。',
 '原中断记录保留于20261011_legacy4_fp32_deploy_v2；本目录是完整结果。',
 'counter_correction.json保存原/新源码哈希及唯一改动。每个原始JSON保存checkpoint哈希及适配说明。', '',
 'summary_reviewed.csv保存本组完整精度汇总。main_table_12models.csv合并前两组，',
 '包含qualification列，保留模型可信度及训练条件限定；不能忽略这些标注。', '',
 f'服务器结果目录：{root.resolve()}。']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print('\n'.join(lines[:16]))
