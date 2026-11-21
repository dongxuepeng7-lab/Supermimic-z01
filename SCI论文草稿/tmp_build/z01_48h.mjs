import fs from "node:fs/promises";
import path from "node:path";
import {createRequire} from "node:module";
import {pathToFileURL} from "node:url";
const resolvePackage = createRequire(path.join(process.argv[2], "package.json")).resolve;
const {Workbook, SpreadsheetFile} = await import(pathToFileURL(resolvePackage("@oai/artifact-tool")));
const book = Workbook.create();
const views = [
  {name:"三机排程", title:"Z01 · 48 小时三机实验安排", subtitle:"从准备开始计时。各阶段为预算；第 4 小时按本机实测吞吐调整。", widths:[13,34,34,36,44], headers:["时间","RTX 4090","RTX 5080","RTX 3090 / 本机 CPU","放行条件与交付"], rows:[
    ["0–4 h","复现 Z01 已成功的平地动作；跑 A0 短训练、保存和重载。","相同版本与精度跑 A 短训练；核对参数实际生效，测显存与稳态步速。","按完整会话划分辨识、校准、测试日志。核对动作接口、单位、采样时间和延迟。","锁定动作、预算、初始化、评估清单及配置。若两种参数在同一指令下输出完全相同，先修参数接线。"],
    ["4–20 h","种子 1 跑 A0，随后同卡跑 A；按相同步数存档。","种子 2 跑 A，随后同卡跑 A0；用共同可承受的有效批量。","跑留出真机日志开环回放；用开发场景验证仿真评估器与终止原因记录。","先估每张卡在第 36 小时前能否完成一对。来不及则在看测试结果前统一降低预算，或转为并行保一对。"],
    ["20–36 h","完成种子 1 的配对；有等待时备份检查点。","完成种子 2 的配对；失败仅按预设规则恢复。","在冻结清单上逐批评估已完成的配对；主任务畅通且时间允许才接第三个完整种子。","第 36 小时冻结模型。慢任务使用预先规定的最近共同预算检查点；记录每组实际步数。"],
    ["36–44 h","补足共同测试工况；用相同后端和精度。","复核配置与动作跟踪记录。主比较完整后才做可选扩展。","统一评估设置；汇总逐回合完成、有效时长、提前终止和误差。","输出每个训练种子的配对差。确定性完全重复的回合只计一个独立情景。"],
    ["44–48 h","复查 A0 训练、初始权重来源和评估记录。","复查 A 的已生效参数、辨识来源和评估记录。","汇总开环曲线、仿真闭环配对图、压力测试曲线，归档数据索引。","保留负结果与失败。单种子标先导；缺独立硬件日志时只报告仿真敏感性。"]
  ]},
  {name:"实验口径",title:"每个实验能回答什么问题",subtitle:"A0 = 原始标称参数；A = 离线辨识参数。所有数值均待实验。",widths:[19,34,45,43,38],headers:["项目","输入与分组","执行规则","记录与比较","解释范围"],rows:[
    ["训练主比较","同一 Z01 基础版 SuperMimic；A0 与 A 仅替换共有执行器实现中的参数值。","同一动作库、奖励、网络、仿真步数、模型与策略更新次数、域随机化偏移和起始权重。两组各自采集仿真转移并训练世界模型和策略。","保存配置差异、实际参数、控制周期、训练种子、检查点与交互步数。热启动时标记初始权重来源。","测整套辨识参数的净收益；单参数作用需要额外消融。"],
    ["参数验收","先选可辨识的关节摩擦、阻尼、指令延迟。前馈项仅在预知输入及辨识依据齐全时启用。","区分关节摩擦与足地摩擦。延迟按时间换算并核对作用环节。辨明前馈指令、增益、偏置和时变残差。","相同初态和指令比较有效力矩、关节响应起点与轨迹；记录饱和及裁剪。","新加延迟队列等结构需两组共享；训练及评估不能输入未来实测力矩。"],
    ["硬件开环回放","独立留出的完整 Z01 日志；两组物理模型回放同一初态、因果历史和指令。","起点后不以实测状态重置。评估 1、5、10 控制步及毫秒数，逐会话配对。只有单关节数据时，按相应试验边界回放。","关节角和速度 RMSE、有效窗口数、必要时响应起点误差。另有足够状态与动作时再比较两组学习出的世界模型。","物理模型回放验证参数对真实记录的预测力；学习模型需要自己的独立预测结果。"],
    ["仿真闭环","冻结策略，放回同一独立物理仿真评估器。测原始标称、共同辨识参考、训练未见的合理参数组合。","测试清单锁定动作、初态和扰动。范围据辨识不确定性或预先说明的工程范围设置。优先已跑通的含换重心或落脚动作。","每组完整完成数 / 总回合、提前终止、有效时长；同一动作关节跟踪 RMSE。行走有参考速度时加速度跟踪 RMSE。","按训练种子报告 A 减 A0 的完成率百分点、A0 减 A 的误差及压力增大后的退化。策略鲁棒性结论限于测试工况。"],
    ["可选扩展","主比较与回放完成、有完整代码和配对预算后再选一项。","可用同一辨识执行器环境，比较匹配输入和容量的普通世界模型与显式执行器世界模型。未收敛的视觉项目单列。","只做单组演示可以记录工程进展；单列不同项目、算法、动作和输入。","独立留出日志或配对训练不足时，优先补齐主证据。"]
  ]}
];
for (const v of views) {
  const s = book.worksheets.add(v.name);
  for (const [i,w] of v.widths.entries()) s.getRangeByIndexes(0,i,1,1).format.columnWidth = w*1.35;
  for (const [i,t] of [v.title,v.subtitle].entries()) {
    const r=s.getRangeByIndexes(i,0,1,5); r.merge(); r.values=[[t]];
    r.format.fill=i===0?"#18374B":"#E7F0F1";r.format.font.color=i===0?"#FFFFFF":"#194956";r.format.font.size=i===0?16:11;r.format.rowHeight=i===0?36:26;
  }
  const h=s.getRange("A3:E3");h.values=[v.headers];h.format.fill="#286365";h.format.font.color="#FFFFFF";h.format.font.bold=true;h.format.rowHeight=27;
  const data=s.getRangeByIndexes(3,0,v.rows.length,5);data.values=v.rows;data.format.wrapText=true;data.format.verticalAlignment="center";data.format.font.size=11;data.format.rowHeight=100;
  v.rows.forEach((_,i)=>{if (i%2===0) s.getRangeByIndexes(3+i,0,1,5).format.fill="#F0F5F5";});s.freezePanes.freezeRows(3);
  if(s.getRange("A4").values[0][0]!==v.rows[0][0]) throw Error("schedule validation");
}
book.recalculate();
for(const v of views){
 const p=await book.render({sheetName:v.name,range:"A1:E8",scale:0.88,format:"png"});
 await fs.writeFile(path.resolve("tmp_build",v.name+"48.png"),new Uint8Array(await p.arrayBuffer()));
}
const file=await SpreadsheetFile.exportXlsx(book);
await file.save(path.resolve("RA-L_Actuator_World_Model","Z01_三机48小时实验安排.xlsx"));
