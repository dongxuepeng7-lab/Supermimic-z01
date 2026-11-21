import fs from "node:fs/promises";
import { createRequire } from "node:module";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const load = createRequire(resolve(process.argv[2], "package.json"));
const { FileBlob, Workbook, SpreadsheetFile } = await import(pathToFileURL(load.resolve("@oai/artifact-tool")));
const prior = await SpreadsheetFile.importXlsx(
  await FileBlob.load(resolve("RA-L_Actuator_World_Model","实验安排表.xlsx")),
);
const priorSchedule = prior.worksheets.getItem("14天安排");
const priorBudget = prior.worksheets.getItem("运行预算");
const previousStatus = new Map(priorSchedule.getRange("A2:G15").values.map(row => [row[0], row[6]]));
const previousBudgetRows = priorBudget.getRange("A2:F9").values;
const previousInputs = new Map(previousBudgetRows.map(row => [row[0], row.slice(1)]));
const previousReserve = priorBudget.getRange(previousInputs.has("A0") ? "B11" : "B10").values[0][0];
const wb = Workbook.create();
const schedule = wb.worksheets.add("14天安排");
const matrix = wb.worksheets.add("实验矩阵");
const budget = wb.worksheets.add("运行预算");
const gains = wb.worksheets.add("辨识收益");
const status = "未开始";

const days = [
  [1,"P0、P1","实际 Z01 仓库、可用日志","定位任务、入口、资产与参考。冻结原始标称执行器配置，盘点指令、位置、速度及时间戳。","保留 Z01 项目版本、标称参数来源、单位、校验值和真机日志清单。","4090 小跑，CPU 查数据",status],
  [2,"P0、P6","D1 仓库盘点","小规模运行 A0 基线并独立重载。锁定跟踪任务含义，测稳定训练吞吐。","确认真实学习更新、检查点和评估可用。记录吞吐、显存和预热。","4090 实测，5080 校验",status],
  [3,"P1、P2、P6","合格的日志清单","按会话切分日志并辨识参数。匹配原始值与辨识值，核对 A0/A 配置。","锁定真机测试会话。配置差异只包含可对照的参数值。记录生效参数和裁剪。","CPU 拟合，GPU 先导",status],
  [4,"P2、P3","D2、D3","接入执行器状态和队列；建立 A+ 对照；从同一物理仿真状态克隆生成成对样本。","动作方向、重置、限幅、梯度与子步对拍通过；正负分支来源可追溯。","5080 验收；4090 跑 A",status],
  [5,"P3、P4","D4 成对样本与模型","先导运行 A+/B/C。独立测试仿真扰动对预留幅度区间或参数组合。","验证 λ_R=0 对应 B。记录方向、幅度、响应起点、误差和有效对数。","5080 模型先导，4090 计时",status],
  [6,"P5、P6、P7","D2、D5 先导","冻结训练分布、共同辨识参考与留出工况、种子、预算、达标线和发散阈值。","锁定评估场景 ID 与真机测试会话。记录窗口候选、温启动与误差容差。","两卡按实测速率排队",status],
  [7,"P4、P6","D6 冻结配置","启动 A0、A、A+、B、C 配对训练。A0/A 采用相同初始化和预算。","追溯每次训练的实际参数、随机化偏移、步数、种子与参数配置版本。","同一对照的计时用同型号卡",status],
  [8,"P5、P6","可用的 C 组检查点","用独立校准集冻结窗口表；从 C 分叉 D/C-perm 并做一次等预算验收。","共同起点、冻结模型与价值网络；每次更新的短长分布与有效想象步精确相同。","5080 窗口先导；4090 续跑",status],
  [9,"P5、P6","D8 等预算验收","完成各组正式训练并处理失败重跑。仅在资源充足时添加固定短窗或长窗。","目标共 3 个配对种子。记录不足与删失；未过阈值的训练不记为成功。","两卡按预算表执行",status],
  [10,"P7","保存的检查点","A0/A 和各组共用辨识参考与留出物理仿真场景、初态、命令和种子。","逐回合记录场景 ID、完成、提前终止、有效时长和速度误差。","两卡评估",status],
  [11,"P1、P7","锁定的测试集","开环回放同一批真机日志时分别使用标称和辨识参数。测留出仿真扰动对。","保存逐步误差、发散和参考结束先后、有效分母，以及响应方向、幅度和起点。","GPU 仿真，CPU 日志回放",status],
  [12,"P8","D9–D11 输出","按种子和场景统计 A0/A 辨识收益，分别汇总共同参考和留出工况。","填完成率百分点与速度 RMSE 降幅，按会话汇总离线回放，并画误差随步长和训练曲线。","CPU 统计，GPU 留作补测",status],
  [13,"P6、P7、P8","明确的缺口","优先补失败种子、复核窗口或检查回放同步；归档配置、脚本与日志索引。","组间同预算账本可复核。若窗口选择退化，记录并据此调整第三项贡献的表述。","两卡留作补跑",status],
  [14,"P8","D12–D13 复核","写入已完成结果，核对仿真与真机日志证据边界，复查匿名稿和图表。","图表都有来源、单位及分母。待测值保留缺项。策略结论只覆盖仿真。","CPU 整理",status],
];

const experiments = [
  ["E0","A0 ↔ A","估计真机辨识参数对原版 SuperMimic 的贡献","仅替换可对照的执行器数值。初始化、随机化偏移、训练预算相配。评估场景一致。","共同辨识参考与留出仿真工况","完成率增益（百分点）、速度 RMSE 降幅（m/s）、达标步数和时间","P0/P6/P7/P8","必做"],
  ["E1","A+ ↔ B","输入匹配的非结构化模型与显式执行器","相同参数/记忆/目标历史、近似容量、同批数据和固定窗","模拟回放与独立闭环","1/5/10 步误差、完成率、速度 RMSE、参数数","P2/P4/P6/P7","必做"],
  ["E2","B ↔ C","留出工况的响应鲁棒性与响应损失增量","同一正负分支做点值拟合。共享初始化和预算，只切换 λ_R。","留出仿真幅度区间或参数组合","响应方向、幅度、起点、逐步误差、有效配对数和闭环结果","P3/P4/P6/P7","必做"],
  ["E3","C-perm ↔ D","只改变短长窗口在同批起点中的分配","同一 C 检查点，冻结世界模型/价值网络，同起点和更新；直方图精确相同","独立仿真校准及锁定测试","预算账本、回退比例、完成率、速度 RMSE","P5/P6/P7","必做"],
  ["E4","标称／辨识物理回放；B/C 按覆盖","留出真机日志上的预测稳定性","共享已记录初态、指令和因果预热。孤立关节只回放对应通道。","按会话留出的原始硬件日志","逐步 q 与 dq RMSE、1/5/10 步和时间、发散率、有效分母","P1/P7/P8","必做"],
  ["E5","C 固定短/长窗","检查窗口长度的影响","同 C 检查点，固定短或长；同时报告真实预算差异","独立闭环仿真","训练步数、时间、性能","P5/P7","备选"],
  ["E6","C 梯度替换","同前向替换执行器反向近似","共享模型和固定窗口，检查前向相等及受限／不限幅时的导数","模拟回放与独立闭环","前向差、梯度特性、策略性能","P2/P4/P7","备选"],
  ["E7","真实成对数据扩展","已核实的硬件扰动对作为响应监督","拟合、校准、测试的硬件对按独立会话分割","仅在确有真实配对时运行","符号与幅度、跨步响应、离线误差与仿真性能","P1/P4/P5","有数据再做"],
];

function writeTable(sheet, headers, rows, widths, rowHeight) {
  const last = String.fromCharCode(64 + headers.length);
  const end = rows.length + 1;
  sheet.showGridLines = false;
  sheet.getRange(`A1:${last}${end}`).values = [headers, ...rows];
  const area = sheet.getRange(`A1:${last}${end}`);
  area.format.font.name = "Microsoft YaHei";
  area.format.font.size = 10;
  area.format.verticalAlignment = "center";
  area.format.wrapText = true;
  area.format.borders = {preset:"all",style:"thin",color:"#D9E2EC"};
  area.format.rowHeight = rowHeight;
  for (let row = 2; row <= end; row += 2) sheet.getRange(`A${row}:${last}${row}`).format.fill = "#EFF5FA";
  widths.forEach((width, i) => {
    const col = String.fromCharCode(65+i);
    sheet.getRange(`${col}:${col}`).format.columnWidthPx = width;
  });
  const header = sheet.getRange(`A1:${last}1`);
  header.format.fill = "#193853";
  header.format.font.color = "#FFFFFF";
  header.format.font.bold = true;
  header.format.rowHeight = 28;
  sheet.freezePanes.freezeRows(1);
}

writeTable(schedule,["第几天","提示词","前置条件","工作内容","验收产物","设备安排","状态"],days.map(row=>[...row.slice(0,6),previousStatus.get(row[0])??status]),[60,90,145,310,315,170,90],53);
schedule.getRange("G2:G15").format.fill = "#E7F2FC";
schedule.getRange("G2:G15").dataValidation = {rule:{type:"list",values:["未开始","进行中","已完成","待补","阻塞"]}};
writeTable(matrix,["实验","组别","问题","对照约束","数据来源","指标与交付物","提示词","优先级"],experiments,[65,110,195,310,215,248,110,85],50);

const jobs = [
  ["A0",3,null,null,"原始标称参数基线。与 A 同卡做成对计时"],
  ["A",3,null,null,"辨识参数基线。与 A0 按同种子配对"],
  ["A+",3,null,null,"执行器输入和容量匹配"],
  ["B",3,null,null,"与 A+/C 同卡形成配对"],
  ["C",3,null,null,"响应监督共用训练数据"],
  ["C-perm",3,null,null,"在同一型号卡上与 D 配对"],
  ["D",3,null,null,"窗口分配，使用冻结的 C 检查点"],
  ["配对额外补跑",0,null,null,"按先导结果改为实际计划"],
];
writeTable(budget,["训练组","计划种子数","4090 分配数","5080 分配数","4090 单次小时","5080 单次小时","4090 预算小时","5080 预算小时","检查与说明"],jobs.map(([name,count])=>[name,...(previousInputs.get(name)??[count,null,null,null,null]),null,null,null]),[125,100,105,105,110,110,115,115,320],31);
jobs.forEach((job,i)=>{
  const r=i+2;
  budget.getRange(`G${r}:I${r}`).formulas=[[
    `=IF(COUNT(C${r})=0,"",IF(C${r}=0,0,IF(COUNT(E${r})=0,"",C${r}*E${r})))`,
    `=IF(COUNT(D${r})=0,"",IF(D${r}=0,0,IF(COUNT(F${r})=0,"",D${r}*F${r})))`,
    `=IF(COUNT(C${r}:D${r})<2,"待分配",IF(SUM(C${r}:D${r})=B${r},"${job[4]}","设备分配与计划种子数不一致"))`
  ]];
});
budget.getRange("B2:F9").format.fill="#E7F2FC";
budget.getRange("E2:H9").setNumberFormat("0.0");
budget.getRange("A11:I11").values=[["备用小时",previousReserve,null,null,null,null,null,null,"配置前填写。未使用的卡可以填 0。"]];
budget.getRange("B11").format.fill="#E7F2FC";
budget.getRange("A12").values=[["正式预算合计"]];
budget.getRange("G12:H12").formulas=[["=IF(COUNT(G2:G9)<ROWS(G2:G9),\"\",SUM(G2:G9))","=IF(COUNT(H2:H9)<ROWS(H2:H9),\"\",SUM(H2:H9))"]];
budget.getRange("A13").values=[["含预留 GPU 小时"]];
budget.getRange("B13").formulas=[["=IF(COUNT(B11,G12,H12)<3,\"\",SUM(B11,G12:H12))"]];
budget.getRange("A14:I14").values=[["计时规则",null,null,null,null,null,null,null,"填入先导实测耗时。A0/A 的同种子运行使用同型号 GPU。"]];
budget.getRange("I11:I14").format.wrapText=true;
budget.getRange("A11:I14").format.rowHeight=32;
budget.getRange("A14:I14").format.rowHeight=50;

const cases = ["辨识参考", "留出工况"];
const gainRows = cases.flatMap(condition => [1,2,3].map(seed => [condition,seed,null,null,null,null,null,null,null]));
writeTable(gains,["测试工况","配对种子","共同回合数","A0 完成数","A 完成数","A0 RMSE (m/s)","A RMSE (m/s)","完成率增益 (百分点)","RMSE 降幅 (m/s)"],gainRows,[125,96,108,108,108,140,140,168,157],32);
gains.getRange("B2:G7").format.fill="#E7F2FC";
gains.getRange("F2:G7").setNumberFormat("0.000");
gains.getRange("H2:H7").setNumberFormat("0.0");
gains.getRange("I2:I7").setNumberFormat("0.000");
gainRows.forEach((_,i)=>{
  const r=i+2;
  gains.getRange(`H${r}:I${r}`).formulas=[[
    `=IF(COUNT(C${r}:E${r})<3,"",IF(AND(C${r}>0,D${r}>=0,E${r}>=0,D${r}<=C${r},E${r}<=C${r}),100*(E${r}-D${r})/C${r},"核对回合数"))`,
    `=IF(COUNT(F${r}:G${r})<2,"",IF(AND(F${r}>=0,G${r}>=0),F${r}-G${r},"核对 RMSE"))`,
  ]];
});
gains.getRange("A9").values=[["种子配对均值"]];
for (const [i,condition] of cases.entries()) {
  const first = 2+i*3;
  const last = first+2;
  const row = 10+i;
  gains.getRange(`A${row}`).values=[[condition]];
  gains.getRange(`H${row}:I${row}`).formulas=[[
    `=IF(COUNT(H${first}:H${last})=0,"",AVERAGE(H${first}:H${last}))`,
    `=IF(COUNT(I${first}:I${last})=0,"",AVERAGE(I${first}:I${last}))`
  ]];
}
gains.getRange("H10:H11").setNumberFormat("0.0");
gains.getRange("I10:I11").setNumberFormat("0.000");
gains.getRange("A13").values=[["填表口径"]];
gains.getRange("B13:I13").merge();
gains.getRange("B13").values=[["蓝色单元格填写真实记录，可把 1–3 改为实际种子。每行按完全相同的测试场景与回合配对。完成率增益＝A−A0，速度 RMSE 降幅＝A0−A，正值表示 A 改善。留出工况按预先固定的规则汇总。"]];
gains.getRange("B13:I13").format.wrapText=true;
gains.getRange("A13:I13").format.rowHeight=45;

const budgetInputs = budget.getRange("C2:F2").values;
budget.getRange("C2:F2").values=[[2,1,1.5,2]];
gains.getRange("C2:G2").values=[[20,12,15,0.17,0.13]];
wb.recalculate();
const [hours4090,hours5080] = budget.getRange("G2:H2").values[0];
if (hours4090 !== 3 || hours5080 !== 2) throw new Error("Budget formula verification failed");
const [completionGain,rmseGain] = gains.getRange("H2:I2").values[0];
if (completionGain !== 15 || Math.abs(rmseGain-0.04)>1e-9) throw new Error("Calibration formula verification failed");
budget.getRange("C2:F2").values=budgetInputs;
gains.getRange("C2:G2").values=[[null,null,null,null,null]];

wb.recalculate();
console.log((await wb.inspect({kind:"table",range:"运行预算!A1:I4",tableMaxRows:4,tableMaxCols:9})).ndjson);
for (const [sheet,range] of [[schedule,"A1:G15"],[matrix,"A1:H9"],[budget,"A1:I14"],[gains,"A1:I13"]]) {
  const blob = await wb.render({sheetName:sheet.name,range,scale:1,format:"png"});
  await fs.writeFile(resolve("tmp_build", `ral-${sheet.name}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const out=await SpreadsheetFile.exportXlsx(wb);
await out.save(resolve("tmp_build","实验安排表_验证更新.xlsx"));
