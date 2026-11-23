Z01 RAL 中英文修订稿与可复核源码 — 2026-11-23

1. 交付范围
manuscript_en.tex / body_en.tex：英文论文，本次交付 PDF 共 8 页。
manuscript_zh.tex / body_zh.tex：逐节完整中文译稿，本次交付 PDF 共 12 页。
中文稿采用单栏阅读排版；参考文献保留原始英文题名以便检索。
图内保留英文标注，中文图注解释变量、坐标、颜色和实验范围。
两个版本共用 table_*.tex、references.bib 与 figures/，避免数值漂移。
源码包内 pdf/ 目录另附本次成稿的双语 PDF。
作者、单位与致谢未编造；英文作者栏保持空白。

2. 本次修订
纳入 online_night01 的 4090/5080 共 31 对种子、62 个模型的回传记录，
以及 3090 的旧模型随机化评估和完整响应后端检查。
当前主表采用在线 A+/B 的非零实测完成率。早期冻结数据阶段实际未完成
任何片段的结果保留于正文，不再重复排列整表 0.00；没有把失败结果改成
理论预测。未执行的 D/C-perm 明确标注“未运行”。
整合 lilun 版中可成立的预算计算、扰动范数、有限差分分支误差界、噪声
方差以及有条件的损失最优性关系。数学推导均明确假设与结论范围，
不推导不存在依据的成功率、RMSE、显著性或训练时间。
保留用户指定的整体框架图，并在图注区分已实现部分与待验证扩展。
统一表格标题字体；添加全部 31 对种子的预测—控制散点图。

3. 科学结论范围
这是一份基于现有证据完成的稿件整理版本，不表示完整研究假设已获验证。
当前数据支持部分前向预测改善，尚不支持稳定的控制优势。
5000 次更新模型的回传包包含元数据，没有实际权重；文中明确披露此限制。
3090 随机化评估使用旧 1000 次更新模型，不与新 5000 次更新模型合并。
响应后端检查未通过；在线响应监督 C、D/C-perm、锁定测试、实机全身迁移
和 PPO 效率对照没有被报告为已完成结果。保留所有完整负结果。

4. 文献
references.bib 包含 18 条引文，优先采用 ACM TOG、Science Robotics、
RSS、ICLR、ICML、CoRL、NeurIPS、AAAI、IROS 的正式记录。
reference_audit.json 逐条记录核验来源和特殊说明。
SuperMimic 的 CoRL 2026 标注来自作者项目页给出的会议引用；本次未独立
确认正式论文集卷页，因此没有编造卷页，也不将它计作出版商记录已核验。
部分 CoRL 会议召开年与论文集出版年不同，BibTeX 采用所引论文集年份。
不再保留原稿中无法核实正式发表状态的 accepted / arXiv-only 表述。

5. 编译
需要 Python 3，以及 Tectonic（本次使用 0.17.0）或 XeLaTeX + BibTeX。
中文需要系统字体 Liberation Serif、Noto Serif CJK SC、Noto Sans CJK SC、
Noto Sans Mono CJK SC。Ubuntu 可安装 fonts-liberation2 与 fonts-noto-cjk。
首次 Tectonic 编译可能下载标准 TeX 包；该步骤需要网络。
在本目录运行：
    python3 build.py
或分别运行：
    tectonic --keep-logs --keep-intermediates manuscript_en.tex
    tectonic --keep-logs --keep-intermediates manuscript_zh.tex
生成 manuscript_en.pdf 与 manuscript_zh.pdf。
包中保留 .bbl 及自带的 ieeeconf.cls；IEEEtran.bst 为标准 TeX 依赖。
在 Overleaf 使用 XeLaTeX，并确保相同字体可用；字体不同可能改变分页。

6. 可复核材料
evidence/online_metadata/：62 个模型的配对预测诊断和 124 份策略评估 JSON，
直接复制自回传记录；不含大模型、动作库或锁定测试数组。
evidence/table_online_values.json：主表未舍入统计。
evidence/*review*.json：各轮记录审核摘要，按轮次保留。
evidence/params.csv：启用的关节参数。
evidence/archived_replay_metrics.csv：归档关节回放表原始汇总值。
evidence/manuscript_lilun_20261122.json：既有理论附录的计算来源。
evidence/reference_metadata/：部分 DOI 的 Crossref 出版元数据。
evidence/input_manifest.json：打包原始元数据的 SHA-256 与来源相对路径。
validate.py：从包内原始 JSON 重算当前完成率、校准指标并检查双语一致性。
    python3 validate.py
build_assets.py：从审核摘要重绘描述性散点图，需 matplotlib。
    python3 build_assets.py
其余两个矢量图直接随包提供。关节回放图仅处理字体嵌入，未改变曲线数据。
validation_report.json：本次数值、编译与版面检查记录。

7. 排版与提交
英文使用原项目 IEEE conference 类和 10 pt 正文；含文献共 8 页。
RA-L 作者说明：https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/
该页规定论文至多 8 页，参考文献及附录计入；不能用额外文本补充材料
绕开页数限制。理论已整合在主文，中文译稿仅供阅读与修改，不是英文
投稿的额外补充论文。作者栏及投稿系统所需信息应由实际作者填写。
