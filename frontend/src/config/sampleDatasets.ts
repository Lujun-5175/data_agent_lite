export interface SampleDataset {
  id: string;
  name: string;
  filename: string;
  description: string;
  rowCount: number;
  columnCount: number;
  path: string;
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
  },
  {
    id: 'students',
    name: '学生成绩',
    filename: 'student_scores.csv',
    description: '探索学习时长、出勤率与期中/期末成绩之间的关系。',
    rowCount: 300,
    columnCount: 9,
    path: '/sample_data/student_scores.csv',
  },
  {
    id: 'behavior',
    name: '用户行为',
    filename: 'user_behavior.csv',
    description: '比较 A/B 组转化率、渠道来源、设备类型和用户活跃分层。',
    rowCount: 1000,
    columnCount: 9,
    path: '/sample_data/user_behavior.csv',
  },
];
