export interface SampleDataset {
  id: string;
  name: string;
  filename: string;
  description: string;
  rowCount: number;
  columnCount: number;
  path: string;
  prompts: string[];
}

export const SAMPLE_DATASETS: SampleDataset[] = [
  {
    id: 'sales',
    name: '销售数据',
    filename: 'sales_data.csv',
    description: '分析不同区域、品类、客户分层和渠道下的销售趋势与收入差异。',
    rowCount: 800,
    columnCount: 10,
    path: '/sample_data/sales_data.csv',
    prompts: [
      '每月销售额趋势是什么？请画一张折线图。',
      '哪个商品品类收入最高？请按区域对比。',
      '比较线上和线下渠道的 total_amount，做一个 t 检验。',
    ],
  },
  {
    id: 'students',
    name: '学生成绩',
    filename: 'student_scores.csv',
    description: '探索学习时长、出勤率与期中/期末成绩之间的关系。',
    rowCount: 300,
    columnCount: 9,
    path: '/sample_data/student_scores.csv',
    prompts: [
      'study_hours 和 final_score 的相关性是多少？',
      '用 study_hours 和 attendance_rate 预测 final_score，跑一个线性回归。',
      '男生和女生的成绩是否有显著差异？请做 t 检验。',
    ],
  },
  {
    id: 'behavior',
    name: '用户行为',
    filename: 'user_behavior.csv',
    description: '比较 A/B 组转化率、渠道来源、设备类型和用户活跃分层。',
    rowCount: 1000,
    columnCount: 9,
    path: '/sample_data/user_behavior.csv',
    prompts: [
      '比较 ab_group A 和 B 的 conversion_flag 转化率，并做卡方检验。',
      '哪个 channel_source 的转化率最高？',
      '按 session_count 对用户分层，并可视化分布。',
    ],
  },
];
