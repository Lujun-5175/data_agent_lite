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
    name: 'Sales Data',
    filename: 'sales_data.csv',
    description: 'Analyze sales trends and revenue differences across regions, categories, customer segments, and channels.',
    rowCount: 800,
    columnCount: 10,
    path: '/sample_data/sales_data.csv',
  },
  {
    id: 'students',
    name: 'Student Scores',
    filename: 'student_scores.csv',
    description: 'Explore the relationship between study hours, attendance rate, and midterm/final exam scores.',
    rowCount: 300,
    columnCount: 9,
    path: '/sample_data/student_scores.csv',
  },
  {
    id: 'behavior',
    name: 'User Behavior',
    filename: 'user_behavior.csv',
    description: 'Compare A/B group conversion rates, channel sources, device types, and user activity segments.',
    rowCount: 1000,
    columnCount: 9,
    path: '/sample_data/user_behavior.csv',
  },
];
